#!/bin/bash

export SATURN_TOKEN=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiYXRsYXMiLCJzYXR1cm4tYXV0aC1wcm94eSJdLCJpc3MiOiJhdGxhcyIsInN1YiI6IjZjMzExOTQ4ZjliMzQwNmQ4ODYwNGRiYzlkMWM5ZWU2IiwiaXNfcmVmcmVzaCI6ZmFsc2UsInVzZXJfaWQiOiI3NDlkM2E4NTY4MzE0NGEwODU0OTk1NDQxNzNlYWFkZiJ9.pMlZb_Bwujh6CojRN6iqmnFen1TV0XxLafD-cpjd80pLzu5zYHCnW0POLEwv9xpn4klzzxVREYxOQIimxhb0oVdgp9c0lbkk971C8drhX3qlCHpUoWc9dE5HihQ4_lO0cU79o-GONxorjvPU4hFozWqMGmVxCdAwhKB_xkaW2S_rb9XIDFVK_tYgIl1rQ0TbFYLY97sxIGsZSHXfmz39y4ko9kwsRgG0SKyQo0I_NVMoZ6EjrKabGWsECP99aAoQJYcUihKs7X47RFKXoOqjb1cjDArqvEuDwoQVfJ7EO8_9Igogi2BpsJb_CB5md9AdjJOPtJQtg-s9m688c-qSpEQYBuYupNuVVwhAVWoxScuBZS1F22VJHM1krwFtPnxV3Xc94ltcBeS09Pnpi7_fbV_QFOkIDNs0TasikrtTs_GGD2mhpKoaVjZRjjMNmCJ0ypGYB8N9vEc_8rHo1L7UAIPY0Zyhek38zFVkZVVqoBKV_kJRGK5f9U-pkLFlUhAE20FuEU5T1RO1SCxt44Y1SUAplhRllrpZR01XPYel2L-amj0XovvoAZsD5OF_nNVP60kVHy3IB8PttuxUwMUChqvimqQkgcXLEMQRdOvO0uxDTqaH2zY55rckagFjEvS28QEbuTDjTwfLZakVV0zmK8gjyETnV9-jxBjfi3aReLc
pip install saturn-client

# --- Run fold 0
fold0_yaml=/home/jovyan/workspace/3D-DINO/task3_fold0_mimic_training.yaml
sc apply ${fold0_yaml} --start

# fold0_yaml=/home/jovyan/workspace/3D-DINO/task1_fold0_mimic_training.yaml
# sc apply ${fold0_yaml} --start

# # --- Run fold 1
fold1_yaml=/home/jovyan/workspace/3D-DINO/task3_fold1_mimic_training.yaml
sc apply ${fold1_yaml} --start

# fold1_yaml=/home/jovyan/workspace/3D-DINO/task1_fold1_mimic_training.yaml
# sc apply ${fold1_yaml} --start

# --- Run fold 2
fold2_yaml=/home/jovyan/workspace/3D-DINO/task3_fold2_mimic_training.yaml
sc apply ${fold2_yaml} --start

# fold2_yaml=/home/jovyan/workspace/3D-DINO/task1_fold2_mimic_training.yaml
# sc apply ${fold2_yaml} --start

# --- Run fold 3
fold3_yaml=/home/jovyan/workspace/3D-DINO/task3_fold3_mimic_training.yaml
sc apply ${fold3_yaml} --start

# fold3_yaml=/home/jovyan/workspace/3D-DINO/task1_fold3_mimic_training.yaml
# sc apply ${fold3_yaml} --start

# # --- Run fold 4
fold4_yaml=/home/jovyan/workspace/3D-DINO/task3_fold4_mimic_training.yaml
sc apply ${fold4_yaml} --start

# fold4_yaml=/home/jovyan/workspace/3D-DINO/task1_fold4_mimic_training.yaml
# sc apply ${fold4_yaml} --start