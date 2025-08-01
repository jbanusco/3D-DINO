# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import torch
import random
from multiprocessing import Value
import math

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove

    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack([s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }



class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h > self.height:
            h -= 1
        while w > self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                print(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        collated_batch["masks_enc"] = collated_masks_enc
        collated_batch["masks_pred"] = collated_masks_pred

        return collated_batch


class MaskCollator3D(object):

    def __init__(
            self,
            input_size=(224, 224, 224),  # Now expects (D, H, W)
            patch_size=16,
            enc_mask_scale=(0.2, 0.8),
            pred_mask_scale=(0.2, 0.8),
            aspect_ratio=(0.3, 3.0),
            depth_ratio=(0.3, 3.0),  # New parameter for depth aspect ratio
            nenc=1,
            npred=2,
            min_keep=8,  # Increased for 3D
            allow_overlap_prob=0.2
    ):
        super(MaskCollator3D, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        elif len(input_size) == 1:
            input_size = input_size * 3
        elif len(input_size) == 2:
            # If 2D size given, assume same for depth
            input_size = (input_size[0], input_size[0], input_size[1])

        self.patch_size = patch_size

        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.depth = input_size[2] // patch_size

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.depth_ratio = depth_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap_prob = allow_overlap_prob
        self.allow_overlap = False
        self._itr_counter = Value('i', -1)
        self._p_scale = 0.0

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale, depth_ratio_scale, register_scale=False):
        _rand = torch.rand(4, generator=generator)  # Need 3 random numbers for 3D

        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand[0].item() * (max_s - min_s)

        if register_scale:
            self._p_scale = mask_scale

        max_keep = int(self.depth * self.height * self.width * mask_scale)

        # -- Sample block aspect-ratios
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand[1].item() * (max_ar - min_ar)

        min_dr, max_dr = depth_ratio_scale
        depth_ratio = min_dr + _rand[2].item() * (max_dr - min_dr)

        self.allow_overlap = self.allow_overlap_prob > _rand[3].item()

        # -- Compute block dimensions (given scale and aspect-ratios)
        # For 3D: volume = h * w * d = max_keep
        # w/h = aspect_ratio, d/h = depth_ratio
        # So: w = aspect_ratio * h, d = depth_ratio * h
        # Volume: h * (aspect_ratio * h) * (depth_ratio * h) = max_keep
        # h^3 = max_keep / (aspect_ratio * depth_ratio)

        h = int(round((max_keep / (aspect_ratio * depth_ratio)) ** (1 / 3)))
        w = int(round(aspect_ratio * h))
        d = int(round(depth_ratio * h))

        # Ensure dimensions don't exceed limits
        while h > self.height:
            h -= 1
        while w > self.width:
            w -= 1
        while d > self.depth:
            d -= 1

        # Ensure minimum size
        h = max(h, 1)
        w = max(w, 1)
        d = max(d, 1)

        return (h, w, d)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w, d = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left-front corner
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            front = torch.randint(0, self.depth - d + 1, (1,))

            mask = torch.zeros((self.height, self.width, self.depth), dtype=torch.int32)
            mask[top:top + h, left:left + w, front:front + d] = 1

            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            mask_indices = torch.nonzero(mask.flatten())

            # -- If mask too small try again
            valid_mask = len(mask_indices) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')

        mask_indices = mask_indices.squeeze()
        if torch.numel(mask_indices) == 1:
            mask_indices = mask_indices.unsqueeze(0)

        # -- Create complement mask
        mask_complement = torch.ones((self.height, self.width, self.depth), dtype=torch.int32)
        mask_complement[top:top + h, left:left + w, front:front + d] = 0

        return mask_indices, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating 3D volumes into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each volume (w/o seed)
        # 4. sample several pred block locations for each volume (w/o seed)
        # 5. return enc mask and pred mask
        '''

        B = len(batch)
        collated_batch = torch.stack([batch[i][0]["images"] for i in range(B)])

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
            depth_ratio_scale=self.depth_ratio,
            register_scale=True)

        enc_mask_scale = self.enc_mask_scale

        if self._p_scale * 3 > self.enc_mask_scale[0]:
            enc_mask_scale[0] = self._p_scale * 3
        if self._p_scale * 3 > self.enc_mask_scale[1]:
            enc_mask_scale[1] = self._p_scale * 3

        e_size = self._sample_block_size(
            generator=g,
            scale=enc_mask_scale,
            aspect_ratio_scale=(1., 1.),
            depth_ratio_scale=(1., 1.))

        self._p_scale = 0.0  # reset

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width * self.depth
        min_keep_enc = self.height * self.width * self.depth

        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                print(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Truncate masks to minimum size for batching
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)


        #collated_batch["masks_enc"] = collated_masks_enc
        #collated_batch["masks_pred"] = collated_masks_pred

        out = {
            'collated_batch': collated_batch,
            'collated_masks_pred': collated_masks_pred,
            'collated_masks_enc': collated_masks_enc,
        }

        return out


if __name__ == '__main__':
    import torch

    g = torch.Generator()
    mc = MaskCollator(input_size=(96, 96), patch_size=16, npred=10)
    p_size = mc._sample_block_size(
        generator=g,
        scale=mc.pred_mask_scale,
        aspect_ratio_scale=mc.aspect_ratio)

    for i in range(1000):
        x = mc._sample_block_mask(p_size)[1]
        if not torch.all(x[:,-1]):
            print(i)
            break
    print(x)
    print("Done!")