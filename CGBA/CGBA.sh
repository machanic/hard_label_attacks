#conda activate torch1.7
#cd xxj/attack/TangentAttack-main/

# CGBA CIFAR-10
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch gdas
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch WRN-28-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch WRN-40-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch densenet-bc-L190-k40
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch pyramidnet272
#
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --targeted --load-random-class-image --arch gdas
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --targeted --load-random-class-image --arch WRN-28-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --targeted --load-random-class-image --arch WRN-40-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --targeted --load-random-class-image --arch densenet-bc-L190-k40
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --targeted --load-random-class-image --arch pyramidnet272

# CGBA ImageNet
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch inceptionv3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch inceptionv4
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch senet154
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch resnet101
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch resnext101_64x4d
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch swin_base_patch4_window7_224
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch gcvit_base
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch jx_vit

python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch inceptionv3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch inceptionv4
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch senet154
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch resnet101
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch resnext101_64x4d
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch swin_base_patch4_window7_224
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch gcvit_base
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset ImageNet --targeted --load-random-class-image --arch jx_vit

# CGBA defense
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model adv_train
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model TRADES
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model feature_distillation
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model com_defend
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model feature_scatter
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model jpeg

python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm l2 --defense_eps 3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm linf --defense_eps 4_div_255
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm linf --defense_eps 8_div_255


# CGBA_H CIFAR-10
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch gdas
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch WRN-28-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch WRN-40-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch densenet-bc-L190-k40
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch pyramidnet272
#
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --targeted --load-random-class-image --arch gdas
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --targeted --load-random-class-image --arch WRN-28-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --targeted --load-random-class-image --arch WRN-40-10-drop
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --targeted --load-random-class-image --arch densenet-bc-L190-k40
#python CGBA/attack_gpu.py --gpu 3 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --targeted --load-random-class-image --arch pyramidnet272

# CGBA_H ImageNet
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch inceptionv3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch inceptionv4
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch senet154
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch resnet101
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch resnext101_64x4d
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch swin_base_patch4_window7_224
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch gcvit_base
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch jx_vit

python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch inceptionv3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch inceptionv4
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch senet154
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch resnet101
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch resnext101_64x4d
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch swin_base_patch4_window7_224
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch gcvit_base
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset ImageNet --targeted --load-random-class-image --arch jx_vit

# CGBA_H defense
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model adv_train
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model TRADES
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model feature_distillation
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model com_defend
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model feature_scatter
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --dataset CIFAR-10 --arch resnet-50 --attack_defense --defense_model jpeg

python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm l2 --defense_eps 3
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm linf --defense_eps 4_div_255
python CGBA/attack.py --gpu 0 --norm l2 --init-rnd-adv --attack_method CGBA_H --arch resnet50 --attack_defense --defense_model adv_train_on_ImageNet --defense_norm linf --defense_eps 8_div_255






