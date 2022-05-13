import torch
from torch.autograd import Function
from .util import Dtype, Stream, assertcuda, load_kernel, _kernel_header_blocks, get_threads_and_blocks, DTYPES_FLOAT

## AVG-SEP
class BlockPadAvg(Function):
    @staticmethod
    def forward(ctx, highres, lowres, highres_map, lowres_map, block_idx, block_size, lowres_factor, padding):
        do_avg = 1
        batch_size = block_idx.size(0)
        spatial_size = (block_idx.size(1) * block_size, block_idx.size(2) * block_size)

        ctx.save_for_backward(highres_map, lowres_map, block_idx)
        ctx.block_size = block_size
        ctx.batch_size = batch_size
        ctx.spatial_size = spatial_size
        ctx.lowres_factor = lowres_factor
        ctx.padding = padding
        ctx.do_avg = do_avg

        height, width = spatial_size

        assert assertcuda(highres, dtypes=DTYPES_FLOAT)
        assert assertcuda(lowres, dtypes=DTYPES_FLOAT)
        assert assertcuda(block_idx, dtypes=(torch.int32))
        assert assertcuda(highres_map, dtypes=(torch.int32))
        assert assertcuda(lowres_map, dtypes=(torch.int32))
        assert len(highres_map) + len(lowres_map) == block_idx.numel()

        def repad_func(data_in, data_map, is_highres):
            N, C, H, W = data_in.shape
            fac = 1 if is_highres else lowres_factor
            data_blocksize =  H + 2 * padding
            assert H == block_size//fac, (H, block_size, fac)
            assert W == block_size//fac, (W, block_size, fac)
            assert H == W, (H, W)
            size = (N, C, data_blocksize, data_blocksize)
            data_out = torch.empty(size, device=data_in.device, dtype=data_in.dtype)
            npixels = len(data_map) * data_blocksize ** 2
            if npixels == 0:
                data_out.fill_(0)
            else:
                block, grid = get_threads_and_blocks(npixels, C)

                f = load_kernel('repad_kernel_sep', _repad_kernel_avg_sep, dtype=Dtype(highres),
                         batch_size=batch_size, channels=C, height=height, width=width,
                        block_size=block_size, lowres_factor=lowres_factor, padding=padding, is_highres=int(is_highres), do_avg=int(do_avg))
                f(block=block, grid=grid,
                        args=[highres.data_ptr(), lowres.data_ptr(), 
                            data_map.data_ptr(), data_out.data_ptr(),
                            block_idx.data_ptr(), int(npixels)],
                    stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            return data_out

        highres_out = repad_func(highres, highres_map, 1)
        lowres_out = repad_func(lowres, lowres_map, 0)
        
        del highres, lowres
        return highres_out, lowres_out

    @staticmethod
    def backward(ctx, grad_highres_in, grad_lowres_in):
        grad_highres_in = grad_highres_in.contiguous()
        grad_lowres_in = grad_lowres_in.contiguous()
        highres_map, lowres_map, block_idx = ctx.saved_variables
        block_size = ctx.block_size
        batch_size = ctx.batch_size
        spatial_size = ctx.spatial_size
        lowres_factor = ctx.lowres_factor
        padding = ctx.padding
        do_avg = ctx.do_avg

        assert assertcuda(grad_highres_in, dtypes=DTYPES_FLOAT)
        assert assertcuda(grad_lowres_in, dtypes=DTYPES_FLOAT)

        height, width = spatial_size

        def data_out_func_bw(data_in, data_map, is_highres):
            N, C, H, W = data_in.shape
            fac = 1 if is_highres else lowres_factor
            data_blocksize = H - 2*padding
            assert H-2*padding == block_size//fac, (H, padding, block_size, fac)
            assert W-2*padding == block_size//fac, (W, padding, block_size, fac)
            assert H == W, (H, W)
            size = (N, C, data_blocksize, data_blocksize)
            data_out = torch.zeros(size, device=data_in.device, dtype=data_in.dtype)
            return data_out
        
        grad_highres_out = data_out_func_bw(grad_highres_in, highres_map, 1)
        grad_lowres_out = data_out_func_bw(grad_lowres_in, lowres_map, 0)

        def data_bw_func(grad_highres_out, grad_lowres_out, data_in, data_map, is_highres):
            if len(data_map) == 0:
                return grad_highres_out, grad_lowres_out
            npixels = len(data_map) * data_in.shape[2] ** 2
            channels = data_in.size(1)
            block, grid = get_threads_and_blocks(npixels, channels)
            fac = load_kernel('repad_kernel_sep_bw', _repad_kernel_avg_sep_bw, dtype=Dtype(data_in),
                     batch_size=batch_size, channels=channels, height=height, width=width,
                    block_size=block_size, lowres_factor=lowres_factor, padding=padding, is_highres=int(is_highres), do_avg=int(do_avg))
            fac(block=block, grid=grid,
                    args=[grad_highres_out.data_ptr(), grad_lowres_out.data_ptr(), 
                        data_in.data_ptr(), data_map.data_ptr(),
                        block_idx.data_ptr(), int(npixels)],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            return grad_highres_out, grad_lowres_out

        grad_highres_out, grad_lowres_out = data_bw_func(grad_highres_out, grad_lowres_out, grad_highres_in, highres_map, 1)
        grad_highres_out, grad_lowres_out = data_bw_func(grad_highres_out, grad_lowres_out, grad_lowres_in, lowres_map, 0)
        return grad_highres_out, grad_lowres_out, None, None, None, None, None, None, None




_repad_kernel_avg_sep = _kernel_header_blocks+'''
#define IS_HIGHRES ${is_highres}
#define DO_AVG ${do_avg}
#define PADDING ${padding}

extern "C"
__global__ void repad_kernel_sep(
    const DTYPE* __restrict__  const highres, const DTYPE* __restrict__ const lowres,
    const int* __restrict__ const data_map, DTYPE* __restrict__ const data_out,
    const int* __restrict__ const block_idx, const int npixels) {

const int BS = IS_HIGHRES ? BLOCK_SIZE : BLOCK_SIZE_LOWRES; // block size
const int BS_OUT = BS + 2*PADDING; // block size with padding (block size of output)

CUDA_KERNEL_LOOP(i, npixels){
    // loop over every output pixel (padded blocks)
    const DTYPE* data_in = IS_HIGHRES ? highres : lowres; // input data
    int BS_IN = BS; // block size of input

    const int n_out = i / (BS_OUT*BS_OUT);        // batch  
    const int h_out = (i / BS_OUT) % BS_OUT;      // height
    const int w_out = i % BS_OUT;                  // width

    int n_in = n_out;
    int h_in = h_out - PADDING;
    int w_in = w_out - PADDING;

    // check if this position is in block's padding 
    const bool left = w_out < PADDING;
    const bool right = w_out >= BS + PADDING;
    const bool top = h_out < PADDING;
    const bool bottom = h_out >= BS + PADDING;

    const bool is_pad = left|right|top|bottom;
    bool zero_pad = false;
    bool downscale = false;

    if(is_pad){
        // find position of patch it is in
        const int block_id = data_map[n_out]; // linear patch id
        const int h_grid = (block_id / GRID_W) % GRID_H;
        const int w_grid = block_id % GRID_W;
        
        // check if it is in the side zero-padding
        zero_pad = ((left & w_grid==0) || (right & w_grid==GRID_W-1) || \
                   (top & h_grid==0) || (bottom & h_grid==GRID_H-1));
        if(!zero_pad){
            // pad by copying from neighbour
            
            int block_id_in = block_id;
            block_id_in -= left; // left neighbor
            block_id_in += right; // right neighbor
            block_id_in -= GRID_W*top; // top neighbor
            block_id_in += GRID_W*bottom; // bottom neighbor

            n_in = block_idx[block_id_in];
            h_in = h_in + top*BS - bottom*BS;
            w_in = w_in + left*BS - right*BS;
            
            const bool is_highres_in = n_in >= 0;
            if(is_highres_in){
                if(!IS_HIGHRES){
                    h_in *= LOWRES_FACTOR;
                    w_in *= LOWRES_FACTOR;
                    data_in = highres;
                    BS_IN = BLOCK_SIZE;
                    downscale = true;
                }
            }else{
                n_in += BATCH_SIZE*GRID_H*GRID_W;
                if(IS_HIGHRES){
                    h_in /= LOWRES_FACTOR;
                    w_in /= LOWRES_FACTOR;
                    data_in = lowres;
                    BS_IN = BLOCK_SIZE_LOWRES;
                }
            }
            //assert(n_in >= 0);
            //assert(h_in >= 0);
            //assert(w_in >= 0);
            //assert(h_in < BS_IN);
            //assert(w_in < BS_IN);
        }
    }

    // channel 0 index 
    const int b_in = n_in*BS_IN*BS_IN*CHANNELS + h_in*BS_IN + w_in;
    const int b_out = n_out*BS_OUT*BS_OUT*CHANNELS + h_out*BS_OUT + w_out; 

    CUDA_CHANNEL_LOOP(c){
        for(int v = 0; v<1; v++){
            DTYPE val = 0;
            if(!zero_pad){
                val = data_in[b_in + (c+v*CHANNELS/1)*BS_IN*BS_IN];
                if(DO_AVG && !IS_HIGHRES && downscale){
                    #pragma unroll
                    for(int ky=0; ky<LOWRES_FACTOR; ++ky){
                        for(int kx=0; kx<LOWRES_FACTOR; ++kx){
                            if (kx==0 & ky==0) continue;
                            val += data_in[b_in +  (c+v*CHANNELS)*BS_IN*BS_IN + BS_IN*ky + kx];
                        }
                    }
                    val /= (DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR);
                }
            }
            data_out[b_out +  (c+v*CHANNELS/1)*BS_OUT*BS_OUT] = val;
        }
    }
} // closes kernel_loop
} // closes kernel
'''


_repad_kernel_avg_sep_bw = _kernel_header_blocks+'''
#define IS_HIGHRES ${is_highres}
#define DO_AVG ${do_avg}
#define PADDING ${padding}

extern "C"
__global__ void repad_kernel_sep_bw(
    DTYPE* __restrict__  const grad_highres_out, DTYPE* __restrict__ const grad_lowres_out,
    const DTYPE* __restrict__ const data_in, const int* __restrict__ const data_map,
    const int* __restrict__ const block_idx, const int npixels) {

const int BS = IS_HIGHRES ? BLOCK_SIZE : BLOCK_SIZE_LOWRES; // block size
const int BS_IN = BS + 2*PADDING; // block size with padding (block size of input)

CUDA_KERNEL_LOOP(i, npixels){
    // loop over every input pixel (padded blocks)
    DTYPE* data_out = IS_HIGHRES ? grad_highres_out : grad_lowres_out; // output data
    int BS_OUT = BS; // block size of output

    const int n_in = i / (BS_IN*BS_IN);        // batch  
    const int h_in = (i / BS_IN) % BS_IN;      // height
    const int w_in = i % BS_IN;                  // width

    int n_out = n_in;
    int h_out = h_in - PADDING;
    int w_out = w_in - PADDING;

    // check if this position is in block's padding 
    const bool left = w_in < PADDING;
    const bool right = w_in >= BS + PADDING;
    const bool top = h_in < PADDING;
    const bool bottom = h_in >= BS + PADDING;

    const bool is_pad = left|right|top|bottom;
    bool zero_pad = false;
    bool downscale = false;

    if(is_pad){
        // find position of patch it is in
        const int block_id = data_map[n_in]; // linear patch id
        const int h_grid = (block_id / GRID_W) % GRID_H;
        const int w_grid = block_id % GRID_W;
        
        // check if it is in the side zero-padding
        zero_pad = ((left & w_grid==0) | (right & w_grid==GRID_W-1) | \
                   (top & h_grid==0) | (bottom & h_grid==GRID_H-1));
        if(!zero_pad){
            // pad by copying from neighbour
            
            int block_id_out = block_id;
            block_id_out -= left; // left neighbor
            block_id_out += right; // right neighbor
            block_id_out -= GRID_W*top; // top neighbor
            block_id_out += GRID_W*bottom; // bottom neighbor

            n_out = block_idx[block_id_out];
            h_out = h_out + top*BS - bottom*BS;
            w_out = w_out + left*BS - right*BS;
            
            const bool is_highres_out = n_out >= 0;
            if(is_highres_out){
                if(!IS_HIGHRES){
                    h_out *= LOWRES_FACTOR;
                    w_out *= LOWRES_FACTOR;
                    data_out = grad_highres_out;
                    BS_OUT = BLOCK_SIZE;
                    downscale = true;
                }
            }else{
                n_out += BATCH_SIZE*GRID_H*GRID_W;
                if(IS_HIGHRES){
                    h_out /= LOWRES_FACTOR;
                    w_out /= LOWRES_FACTOR;
                    data_out = grad_lowres_out;
                    BS_OUT = BLOCK_SIZE_LOWRES;
                }
            }
            //assert(n_out >= 0);
            //assert(h_out >= 0);
            //assert(w_out >= 0);
            //assert(h_out < BS_OUT);
            //assert(w_out < BS_OUT);
        }
    }

    // channel 0 index 
    const int b_in = n_in*BS_IN*BS_IN*CHANNELS + h_in*BS_IN + w_in;
    const int b_out = n_out*BS_OUT*BS_OUT*CHANNELS + h_out*BS_OUT + w_out; 

    CUDA_CHANNEL_LOOP(c){
        if(!zero_pad){
            DTYPE val = data_in[b_in + c*BS_IN*BS_IN];

            if(DO_AVG && !IS_HIGHRES && downscale){
                val /= (DTYPE) (LOWRES_FACTOR*LOWRES_FACTOR);
                for(int ky=0; ky<LOWRES_FACTOR; ++ky){
                    for(int kx=0; kx<LOWRES_FACTOR; ++kx){
                         atomicAdd(data_out + b_out + c*BS_OUT*BS_OUT + BS_OUT*ky + kx, val);
                    }
                }
            }else{
                atomicAdd(data_out + b_out + c*BS_OUT*BS_OUT, val);
            }
        }
    }
} // closes kernel_loop
} // closes kernel
'''




class BlockPadAvgSep(Function):
    @staticmethod
    def forward(ctx, highres, lowres, highres_map, lowres_map, block_idx, padding, is_highres):
        do_avg = 1
        assert assertcuda(highres, dtypes=DTYPES_FLOAT)
        assert assertcuda(lowres, dtypes=DTYPES_FLOAT)
        assert assertcuda(block_idx, dtypes=(torch.int32))
        assert assertcuda(highres_map, dtypes=(torch.int32))
        assert assertcuda(lowres_map, dtypes=(torch.int32))
        assert len(highres_map) + len(lowres_map) == block_idx.numel()
        assert highres.shape[2] == highres.shape[3]
        assert lowres.shape[2] == lowres.shape[3]
        assert highres.shape[2] % lowres.shape[2] == 0
        ctx.save_for_backward(highres_map, lowres_map, block_idx)
        ctx.padding = padding
        ctx.do_avg = do_avg
        ctx.is_highres = is_highres

        batch_size, grid_h, grid_w = block_idx.shape
        block_size = highres.shape[2]
        lowres_factor = highres.shape[2]//lowres.shape[2]
        height, width = grid_h * block_size, grid_w * block_size

        ctx.block_size = block_size
        ctx.lowres_factor = lowres_factor

        data_in = highres if is_highres else lowres
        data_map = highres_map if is_highres else lowres_map
        num_blocks, C, side, _ = data_in.shape
        block_size_pad = side + 2 * padding
        size = (num_blocks, C, block_size_pad, block_size_pad)
        data_out = torch.empty(size, device=data_in.device, dtype=data_in.dtype)
        if len(data_map) == 0:
            data_out.fill_(0)
        else:
            npixels = len(data_map) * block_size_pad ** 2
            block, grid = get_threads_and_blocks(npixels, C)

            f = load_kernel('repad_kernel_sep', _repad_kernel_avg_sep, dtype=Dtype(highres),
                        batch_size=batch_size, channels=C, height=height, width=width,
                    block_size=block_size, lowres_factor=lowres_factor, padding=padding, is_highres=int(is_highres), do_avg=int(do_avg))
            f(block=block, grid=grid,
                    args=[highres.data_ptr(), lowres.data_ptr(), 
                        data_map.data_ptr(), data_out.data_ptr(),
                        block_idx.data_ptr(), int(npixels)],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return data_out

    @staticmethod
    def backward(ctx, grad_data_pad):
        grad_highres_in = grad_data_pad.contiguous()
        assert assertcuda(grad_highres_in, dtypes=DTYPES_FLOAT)

        highres_map, lowres_map, block_idx = ctx.saved_variables
        block_size = ctx.block_size
        channels = grad_data_pad.size(1)
        lowres_factor = ctx.lowres_factor
        padding = ctx.padding
        do_avg = ctx.do_avg

        batch_size, grid_h, grid_w = block_idx.shape
        height, width = grid_h * ctx.block_size, grid_w * ctx.block_size

        def create_tensors(data_map, block_size):
            n = max(len(data_map), 1)
            size = (n, channels, block_size, block_size)
            return torch.zeros(size, device=grad_data_pad.device, dtype=grad_data_pad.dtype)
        
        grad_highres = create_tensors(highres_map, block_size)
        grad_lowres = create_tensors(lowres_map, block_size // lowres_factor)

        data_map = highres_map if ctx.is_highres else lowres_map
        if len(data_map) > 0:
            npixels = len(data_map) * grad_data_pad.shape[2] ** 2
            block, grid = get_threads_and_blocks(npixels, channels)
            fac = load_kernel('repad_kernel_sep_bw', _repad_kernel_avg_sep_bw, dtype=Dtype(grad_data_pad),
                        batch_size=batch_size, channels=channels, height=height, width=width,
                    block_size=block_size, lowres_factor=lowres_factor, padding=padding, is_highres=int(ctx.is_highres), do_avg=int(do_avg))
            fac(block=block, grid=grid,
                    args=[grad_highres.data_ptr(), grad_lowres.data_ptr(), 
                        grad_data_pad.data_ptr(), data_map.data_ptr(),
                        block_idx.data_ptr(), int(npixels)],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_highres, grad_lowres, None, None, None, None, None


