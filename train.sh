CUDA_VISIBLE_DEVICES=1 python main_dino.py --batch_size_per_gpu 16 \
    --epochs 100 \
    --arch vit_tiny \
    --output_dir "runs/glacial_c012345678_vit_tiny_16_nonorm2" \
    --data_path "/home/jperez/data/HKH/processed_L07_2005/train"
    # --data_path /home/jperez/data/alexbrain/train/ff_images \
