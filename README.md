# MIRnet

Source https://github.com/swz30/MIRNet

Paper https://arxiv.org/abs/2003.06792

# Train
```
python  train.py --noise_dir ../image/Noisy/ --gt_dir ../image/Clean/ --image_size 128 --batch_size 1 --save_every 1000 --loss_every 100 -nw 1 -c  -ckpt mir_kpn --model_type KPN
```


# Test 
```
python test_custom_mat.py  -n /mnt/vinai/SIDD/ValidationNoisyBlocksSrgb.mat  -g /mnt/vinai/SIDD/ValidationGtBlocksSrgb.mat  -c -ckpt mir_kpn -m KPN
```

## Requirement 
mpmath
torch_dct
