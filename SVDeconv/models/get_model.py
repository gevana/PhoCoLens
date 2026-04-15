from models.multi_fftlayer_diff import MultiFFTLayer_diff as SVDeconvLayer_diff
# from models.multi_fftlayer_new import MultiFFTLayer_new as SVDeconvLayer
from models.multi_fftlayer import MultiFFTLayer as SVDeconvLayer

from models.fftlayer import FFTLayer
from models.fftlayer_diff import FFTLayer_diff
from models.fftlayer_diff import FFTLayer_dummy
from models.unet_128 import Unet as Unet_128
from models.unet import UNet270480 as Unet_diff

def get_inversion_and_channels(args):
    
    if hasattr(args, "is_svd"):
        is_svd = args.is_svd
    else:
        is_svd = "svd" in args.exp_name
    if hasattr(args, "is_diff"):
        is_diff = args.is_diff
    else:
        is_diff = "diff" in args.exp_name
    if hasattr(args, "is_dummy_wiener"):
        is_dummy_wiener = args.is_dummy_wiener
    else:
        is_dummy_wiener = "dummy_wiener" in args.exp_name
    #is_svd = "svd" in args.exp_name
    #is_diff = "diff" in args.exp_name
    if is_dummy_wiener:
        print("warning: Using dummy Wiener")
        return FFTLayer_dummy, 3


    if is_svd and not is_diff:
        return SVDeconvLayer, 4 if args.load_raw else 3
        # return SVDeconvLayer, 8 if args.load_raw else 6

    elif is_svd and is_diff:
        return SVDeconvLayer_diff, 6
    elif not is_diff:
        return FFTLayer, 3
    else:
        return FFTLayer_diff, 3

def model(args):
    Inversion, in_c = get_inversion_and_channels(args)

    if args.model == "unet-128-pixelshuffle-invert":
        return Unet_128(args, in_c=in_c), Inversion(args)
    elif args.model == "UNet270480":
        return Unet_diff(args, in_c=in_c), Inversion(args)