"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from .StarGAN_v2 import StarGAN_v2
import argparse
from .utils import *
import time


"""parsing and configuration"""
def parse_args(args):
    desc = "Tensorflow implementation of StarGAN_v2"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--dataset_path', type=str, help='dataset path')
    
    parser.add_argument('--test_src_dir', type=str, help='test src path')
    parser.add_argument('--test_ref_dir', type=str, help='test ref path')
    
    parser.add_argument('--custom_domain_list', type=str, help='only consider the given domains when testing')
    parser.add_argument('--latent_guided_synthesis', type=str2bool, default=True, help='In test phase, if latent guided synthesis should be performed')
    parser.add_argument('--reference_guided_synthesis', type=str2bool, default=True, help='In test phase, if reference guided synthesis should be performed')
    
    parser.add_argument('--debug_logging', type=str2bool, default=True, help='If debug logging should be performed')
    
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--merge', type=str2bool, default=True, help='In test phase, merge reference-guided image result or not')
    parser.add_argument('--merge_size', type=int, default=0, help='merge size matching number')
    parser.add_argument('--dataset', type=str, default='celeba_hq_gender', help='dataset_name')
    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--ds_iter', type=int, default=100000, help='Number of iterations to optimize diversity sensitive loss')

    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size')  # each gpu
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of ckpt_save_freq')
    parser.add_argument('--num_style', type=int, default=5, help='Number of generated images per domain during sampling')

    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay value')

    parser.add_argument('--adv_weight', type=float, default=1, help='The weight of Adversarial loss')
    parser.add_argument('--sty_weight', type=float, default=1, help='Weight for style reconstruction loss')
    parser.add_argument('--ds_weight', type=float, default=1, help='Weight for diversity sensitive loss') # 2 for animal
    parser.add_argument('--cyc_weight', type=float, default=1, help='Weight for cyclic consistency loss')
    parser.add_argument('--r1_weight', type=float, default=1, help='Weight for R1 regularization')

    parser.add_argument('--gan_type', type=str, default='gan-gp', help='gan / lsgan / gan-gp / hinge')
    parser.add_argument('--sn', type=str2bool, default=False, help='using spectral norm')

    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    if args is None:
        return check_args(parser.parse_args())

    return check_args(parser.parse_args(args))


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
#    check_folder(args.checkpoint_dir)

    # --result_dir
#    check_folder(args.result_dir)

    # --result_dir
#    check_folder(args.log_dir)

    # --sample_dir
#    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.iteration >= 1
    except:
        print('number of iterations must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

gan = None

"""main"""
def main(custom_args=None):

    args = parse_args(custom_args)
    
    automatic_gpu_usage(args.debug_logging)

    global gan

    if args.phase == 'train' :
        gan = StarGAN_v2(args)
    
        # build graph
        gan.build_model()

        gan.train()
        print(" [*] Training finished!")
    else:
        
        if not gan:
            print("Model not build yet, building...")
            gan = StarGAN_v2(args)

            # build graph
            gan.build_model()
        else: 
            if args.debug_logging:
                print("Model already build, reload...")
            
            generator_ema = gan.generator_ema
            mapping_network_ema = gan.mapping_network_ema
            style_encoder_ema = gan.style_encoder_ema
            ckpt = gan.ckpt
            manager = gan.manager

            gan = StarGAN_v2(args)

            # build graph
            gan.generator_ema = generator_ema
            gan.mapping_network_ema = mapping_network_ema
            gan.style_encoder_ema = style_encoder_ema
            gan.ckpt = ckpt
            gan.manager = manager

            gan.build_model(reload_checkpoint_only=True)
            
        gan.test(args.merge, args.merge_size)
        if args.debug_logging:
            print(" [*] Test finished!")


if __name__ == '__main__':
    main()
