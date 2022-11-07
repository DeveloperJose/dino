python visualize_attention.py \
    --arch vit_tiny \
    --patch_size 16 \
    --image_path "/home/jperez/data/HKH/processed_L07_2005/train/tiff_140_slice_1.npy" \
    --output_dir "runs/glacial_c012345678_vit_tiny_16_nonorm2/visual/" \
    --pretrained_weights "runs/glacial_c012345678_vit_tiny_16_nonorm2/checkpoint0000.pth" \
    --image_size 512 \
    --threshold 0.1