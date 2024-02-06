运行下面命令
nohup python sphere.py --niter 100000 --gpu 2 > sphere.txt 2>&1 &
nohup python sign_prior_sphere.py --niter 100000 --gpu 2 > sign_prior_sphere.txt 2>&1 &
nohup python prior_sphere_prac.py --niter 100000 --gpu 2 > prior_sphere_prac.txt 2>&1 &
大概一个要跑12小时，把输出的结果发给我
