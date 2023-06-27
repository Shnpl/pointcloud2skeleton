import torch
from torch import Tensor
def square_distance(src:Tensor, dst:Tensor):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    # batchsize,n,_ = src.shape

    # distances = torch.zeros(batchsize, n, n)
    sum_sq_src = torch.sum(src**2,dim=-1).unsqueeze(2)
    sum_sq_dst = torch.sum(dst**2,dim=-1).unsqueeze(1)
    dists = torch.sqrt(sum_sq_src+sum_sq_dst-2*torch.bmm(src,dst.permute(0,2,1)))
    return dists

if __name__ == "__main__":
    x = torch.tensor([[0,0],
                      [0,1],
                      [0,2],
                      [0,3],
                      [0,4]]).unsqueeze(0)
    print(x.shape)
    y = torch.tensor([[0,0],
                      [1,0],
                      [2,0],
                      [3,0],
                      [4,0]]).unsqueeze(0)
    dists = square_distance(x,y)
    print(dists)