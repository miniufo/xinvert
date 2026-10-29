"""Optional float64 CUDA implementation of the closed Li operator.

Topology and Jacobi setup reuse the CPU oracle. No sparse matrix is built.
Iteration vectors stay on device; only the final native solution returns.
"""

import time

import numpy as np

_SOURCE = r"""
typedef INDEX idx;
extern "C" __global__ void rhs(const double* data,double* out,
 const idx* ui,const idx* vi,const double* wu,const double* wv,
 long ny,long nx,long m,long n){
 long t=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(t<ny*nx){long j=t/nx,k=ui[t];if(k>=0)out[k]=wu[j]*data[k];
  k=vi[t];if(k>=0)out[k]=wv[j]*data[k];}
 if(t<n)out[m+t]=0.;
}
__device__ double vertex(long j,long i,const idx* vm,const idx* ui,const idx* vi,
 const double* up,const double* vp,const double* r,double sc,long ny,long nx,long nvx){
 double t=0.; long left=(i+nx-1)%nx; bool px=nvx==nx;
 if(px||i>0){
  if(j>0){long k=ui[(j-1)*nx+left];if(k>=0&&vm[(j-1)*nvx+i]!=vm[j*nvx+i])t+=(-up[j-1]*sc)*r[k];}
  if(j<ny){long k=ui[j*nx+left];if(k>=0&&vm[j*nvx+i]!=vm[(j+1)*nvx+i])t+=(up[j]*sc)*r[k];}
 }
 if(j>0){
  long kl=(px||i>0)?vi[(j-1)*nx+left]:-1;
  long kr=i<nx?vi[(j-1)*nx+i]:-1;
  if(kl>=0&&vm[j*nvx+left]==vm[j*nvx+i])kl=-1;
  if(kr>=0&&vm[j*nvx+i]==vm[j*nvx+(i+1)%nvx])kr=-1;
  double q=vp[j-1];
  if(kl>=0){if(kr>=0&&kr<kl){t+=(-q*sc)*r[kr];kr=-1;}t+=(q*sc)*r[kl];}
  if(kr>=0)t+=(-q*sc)*r[kr];
 }
 return t;
}
extern "C" __global__ void forward(const double* z,double* out,
 const idx* vm,const idx* cm,const idx* ui,const idx* vi,
 const double* up,const double* uc,const double* vp,const double* vc,
 const double* sc,const double* pen,long ny,long nx,long nvx,long npot,long m,long n){
 long t=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(t<ny*nx){long j=t/nx,i=t%nx,k=ui[t];double val;
  if(k>=0){long a=npot+cm[t],b=npot+cm[j*nx+(i+1)%nx];
   if(a>b)val=(-uc[j]*sc[a])*z[a]+(uc[j]*sc[b])*z[b];
   else val=(uc[j]*sc[b])*z[b]+(-uc[j]*sc[a])*z[a];
   a=vm[j*nvx+(i+1)%nvx];b=vm[(j+1)*nvx+(i+1)%nvx];
   if(a!=b){if(a>b){val+=(up[j]*sc[a])*z[a];val+=(-up[j]*sc[b])*z[b];}
    else {val+=(-up[j]*sc[b])*z[b];val+=(up[j]*sc[a])*z[a];}}
   out[k]=val;
  }
  k=vi[t];if(k>=0){long a=npot+cm[t],b=npot+cm[(j+1)*nx+i];
   val=(vc[j]*sc[b])*z[b]+(-vc[j]*sc[a])*z[a];
   a=vm[(j+1)*nvx+i];b=vm[(j+1)*nvx+(i+1)%nvx];
   if(a!=b){if(a>b){val+=(-vp[j]*sc[a])*z[a];val+=(vp[j]*sc[b])*z[b];}
    else {val+=(vp[j]*sc[b])*z[b];val+=(-vp[j]*sc[a])*z[a];}}
   out[k]=val;
  }
 }
 if(t<n)out[m+t]=pen[t]*z[t];
}
extern "C" __global__ void transpose(const double* r,double* out,
 const idx* vm,const idx* cm,const idx* ui,const idx* vi,const unsigned int* counts,
 const double* up,const double* uc,const double* vp,const double* vc,
 const double* sc,const double* pen,long ny,long nx,long nvx,long npot,long m){
 long t=(long)blockIdx.x*blockDim.x+threadIdx.x;
 if(t<ny*nx){long j=t/nx,i=t%nx,ci=cm[t];if(ci>=0){
  long c=npot+ci;double s=sc[c],total=0.;
  long west=ui[j*nx+(i+nx-1)%nx],east=ui[t];if(nvx!=nx&&i==0)west=-1;
  if(west>=0){if(east>=0&&east<west){total+=(-uc[j]*s)*r[east];east=-1;}total+=(uc[j]*s)*r[west];}
  if(east>=0)total+=(-uc[j]*s)*r[east];
  long south=j>0?vi[(j-1)*nx+i]:-1,north=vi[t];
  if(south>=0)total+=(vc[j-1]*s)*r[south];if(north>=0)total+=(-vc[j]*s)*r[north];
  out[c]=total+pen[c]*r[m+c];
 }}
 if(t<(ny+1)*nvx){long j=t/nvx,i=t%nvx,c=vm[t];if(c>=0&&counts[c]==1)
  out[c]=vertex(j,i,vm,ui,vi,up,vp,r,sc[c],ny,nx,nvx)+pen[c]*r[m+c];}
}
extern "C" __global__ void coast_part(const double* r,double* parts,
 const idx* vm,const idx* ui,const idx* vi,const idx* vertices,
 const long long* starts,const long long* ends,const idx* owners,
 const double* up,const double* vp,const double* sc,long ny,long nx,long nvx){
 __shared__ double shared[256];long b=blockIdx.x;int lane=threadIdx.x;double val=0.;
 long k=starts[b]+lane;if(k<ends[b]){long v=vertices[k];
  val=vertex(v/nvx,v%nvx,vm,ui,vi,up,vp,r,sc[owners[b]],ny,nx,nvx);}
 shared[lane]=val;__syncthreads();
 for(int d=128;d>0;d>>=1){if(lane<d)shared[lane]+=shared[lane+d];__syncthreads();}
 if(lane==0)parts[b]=shared[0];
}
extern "C" __global__ void coast_finish(const double* r,double* out,const double* parts,
 const idx* ids,const long long* offsets,const double* pen,long m){
 __shared__ double shared[256];long b=blockIdx.x;int lane=threadIdx.x;double val=0.;
 for(long k=offsets[b]+lane;k<offsets[b+1];k+=256)val+=parts[k];
 shared[lane]=val;__syncthreads();
 for(int d=128;d>0;d>>=1){if(lane<d)shared[lane]+=shared[lane+d];__syncthreads();}
 if(lane==0){long c=ids[b];out[c]=shared[0]+pen[c]*r[m+c];}
}
"""


def require_cupy():
    """Import lazily so CPU-only installations remain supported."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "Li GPU requires optional CuPy matching the installed CUDA toolkit"
        ) from exc
    if not cp.cuda.runtime.getDeviceCount():
        raise RuntimeError("Li GPU requires an available CUDA device")
    return cp


class _GPUOperator:
    """Augmented weighted/preconditioned operator with deterministic gathers."""

    def __init__(self, op):
        cp = self.cp = require_cupy()
        if op.scale is None:
            raise ValueError("prepare the CPU topology before GPU transfer")
        self.n, self.m = op.size, op.data_size
        self.ny, self.nx = op.shape
        self.nvx, self.npot = op.vm.shape[1], op.npsi
        # The disposable RHS doubles as u. Two augmented vectors plus seven
        # unknown vectors conservatively cover the fused LSMR workspace.
        maps = sum(
            a.nbytes
            for a in (
                op.vm,
                op.cm,
                op.ui,
                op.vi,
                op.counts,
                op.vertices,
                op.coast_ids,
                op.offsets,
            )
        )
        self.estimated_bytes = maps + 8 * (2 * (self.m + self.n) + 7 * self.n)
        free = (
            cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        )
        if self.estimated_bytes > free:
            raise MemoryError(
                f"Li GPU estimates {self.estimated_bytes / 2**30:.2f} GiB; "
                f"available {free / 2**30:.2f} GiB. Use a smaller domain or "
                "a device with more memory; precision is not changed automatically."
            )
        self.maps = tuple(cp.asarray(a) for a in (op.vm, op.cm, op.ui, op.vi))
        self.counts = cp.asarray(op.counts)
        self.coeff = tuple(cp.asarray(a) for a in op.coeff)
        self.weights = cp.asarray(op.wu), cp.asarray(op.wv)
        self.scale, self.penalty = cp.asarray(op.scale), cp.asarray(op.penalty)
        self.vertices, self.ids = cp.asarray(op.vertices), cp.asarray(op.coast_ids)
        starts, ends, owners, offsets = [], [], [], [0]
        for b, c in enumerate(op.coast_ids):
            for start in range(int(op.offsets[b]), int(op.offsets[b + 1]), 256):
                starts.append(start)
                ends.append(min(start + 256, int(op.offsets[b + 1])))
                owners.append(c)
            offsets.append(len(starts))
        self.starts = cp.asarray(np.asarray(starts, dtype=np.int64))
        self.ends = cp.asarray(np.asarray(ends, dtype=np.int64))
        self.owners = cp.asarray(np.asarray(owners, dtype=op.vm.dtype))
        self.offsets = cp.asarray(np.asarray(offsets, dtype=np.int64))
        self.parts = cp.empty(len(starts), dtype=cp.float64)
        source = "typedef long long int64_t;\n" + _SOURCE.replace(
            "long long", "int64_t"
        ).replace("long", "int64_t")
        source = source.replace(
            "INDEX", "int" if op.vm.dtype.itemsize == 4 else "int64_t"
        )
        self.module = cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"))
        self.kernels = {
            n: self.module.get_function(n)
            for n in ("rhs", "forward", "transpose", "coast_part", "coast_finish")
        }

    def rhs(self, data):
        """Build augmented RHS on device without a dense host augmentation."""
        # Pad a shorter face vector so its cached block can be reused by LSMR
        # unknown vectors instead of leaving a nearly equal-sized free block.
        uploaded = self.cp.empty(max(self.m, self.n), dtype=self.cp.float64)
        uploaded[: self.m].set(data)
        data = uploaded
        out = self.cp.empty(self.m + self.n, dtype=self.cp.float64)
        size = max(self.ny * self.nx, self.n)
        self.kernels["rhs"](
            ((size + 255) // 256,),
            (256,),
            (
                data,
                out,
                self.maps[2],
                self.maps[3],
                *self.weights,
                *map(np.int64, (self.ny, self.nx, self.m, self.n)),
            ),
        )
        return out

    def matvec(self, z):
        out = self.cp.empty(self.m + self.n, dtype=self.cp.float64)
        size = max(self.ny * self.nx, self.n)
        self.kernels["forward"](
            ((size + 255) // 256,),
            (256,),
            (
                z,
                out,
                *self.maps,
                *self.coeff,
                self.scale,
                self.penalty,
                *map(np.int64, (self.ny, self.nx, self.nvx, self.npot, self.m, self.n)),
            ),
        )
        return out

    def rmatvec(self, r):
        out = self.cp.empty(self.n, dtype=self.cp.float64)
        size = max(self.ny * self.nx, (self.ny + 1) * self.nvx)
        self.kernels["transpose"](
            ((size + 255) // 256,),
            (256,),
            (
                r,
                out,
                *self.maps,
                self.counts,
                *self.coeff,
                self.scale,
                self.penalty,
                *map(np.int64, (self.ny, self.nx, self.nvx, self.npot, self.m)),
            ),
        )
        if len(self.ids):
            self.kernels["coast_part"](
                (len(self.parts),),
                (256,),
                (
                    r,
                    self.parts,
                    self.maps[0],
                    self.maps[2],
                    self.maps[3],
                    self.vertices,
                    self.starts,
                    self.ends,
                    self.owners,
                    self.coeff[0],
                    self.coeff[2],
                    self.scale,
                    *map(np.int64, (self.ny, self.nx, self.nvx)),
                ),
            )
            self.kernels["coast_finish"](
                (len(self.ids),),
                (256,),
                (
                    r,
                    out,
                    self.parts,
                    self.ids,
                    self.offsets,
                    self.penalty,
                    np.int64(self.m),
                ),
            )
        return out


def solve_closed_gpu(
    op,
    data,
    length,
    alpha,
    *,
    maxiter,
    tolerance,
    conlim,
    initial=None,
    solver="lsmr",
    show=False,
):
    """Solve on device, return physical float64 coefficients and CPU diagnostics."""
    if solver != "lsmr":
        raise NotImplementedError("Li GPU currently supports solver='lsmr' only")
    cp = require_cupy()
    from cupyx.scipy.sparse.linalg import LinearOperator

    start = time.perf_counter()
    op.prepare(length, alpha)
    gpu = _GPUOperator(op)
    b = gpu.rhs(data)
    x0 = None
    if initial is not None:
        x0 = cp.asarray(initial)
        x0 /= gpu.scale
        x0 /= length
    op.scale = op.penalty = None
    operator = LinearOperator(
        (gpu.m + gpu.n, gpu.n), dtype=cp.float64, matvec=gpu.matvec, rmatvec=gpu.rmatvec
    )
    result = _lsmr(
        operator,
        b,
        x0=x0,
        atol=tolerance,
        btol=tolerance,
        conlim=conlim,
        maxiter=maxiter,
        overwrite_b=True,
    )
    del b, x0, operator
    z = result[0]
    z *= gpu.scale
    z *= length
    out = cp.asnumpy(z)
    cp.cuda.get_current_stream().synchronize()
    status = {
        "stop_code": int(result[1]),
        "iterations": int(result[2]),
        "converged": result[1] in (0, 1, 2, 4, 5),
        "elapsed_seconds": time.perf_counter() - start,
        "estimated_device_bytes": gpu.estimated_bytes,
    }
    if show:
        print(f"Li GPU: {status}", flush=True)
    op.scale = op.penalty = None
    return out, status


# LSMR scalar recurrence adapted from CuPy 14.2, cupyx/scipy/sparse/linalg/_solve.py.
# Keep the upstream license with this adaptation.
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
_UPDATES = r"""
extern "C" __global__ void lsmr_axpy(double* x,const double* y,double a,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){double t=x[i]*a;x[i]=t+y[i];}
}
extern "C" __global__ void lsmr_update(double* hb,double* h,double* x,
 const double* v,double a,double b,double c,long long n){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){double hi=h[i],q=hb[i]*a;q=q+hi;
  double dx=b*q; x[i]=x[i]+dx; hb[i]=q;
  double hv=hi*c; h[i]=hv+v[i];}
}
"""


def _sym_ortho(a, b):
    """
    A stable implementation of Givens rotation according to
    S.-C. Choi, "Iterative Methods for Singular Linear Equations
      and Least-Squares Problems", Dissertation,
      http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return (np.sign(a), 0, abs(a))
    elif a == 0:
        return (0, np.sign(b), abs(b))
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return (c, s, r)


def _lsmr(
    A,
    b,
    x0=None,
    damp=0.0,
    atol=1e-06,
    btol=1e-06,
    conlim=100000000.0,
    maxiter=None,
    *,
    overwrite_b=False,
):
    """Float64 LSMR with fused vector updates and unchanged stopping tests."""
    cp = require_cupy()
    from cupy import cublas

    update = cp.RawKernel(_UPDATES, "lsmr_update", options=("--fmad=false",))
    axpy = cp.RawKernel(_UPDATES, "lsmr_axpy", options=("--fmad=false",))
    b = b.squeeze()
    matvec = A.matvec
    rmatvec = A.rmatvec
    m, n = A.shape
    minDim = min([m, n])
    if maxiter is None:
        maxiter = minDim * 5
    if x0 is None:
        dtype = cp.result_type(A, b, float)
    else:
        dtype = cp.result_type(A, b, x0, float)
    # solve_closed_gpu owns a disposable RHS; reuse it for bidiagonalization.
    u = b if overwrite_b else b.copy()
    normb = cublas.nrm2(b)
    beta = normb.copy()
    normb = normb.get().item()
    if x0 is None:
        x = cp.zeros((n,), dtype=dtype)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError("x0 has incompatible dimensions")
        x = x0.astype(A.dtype).ravel()
        u -= matvec(x)
        beta = cublas.nrm2(u)
    beta_cpu = beta.get().item()
    v = cp.zeros(n, dtype)
    alpha = cp.zeros((), dtype=beta.dtype)
    alpha_cpu = 0
    if beta_cpu > 0:
        u /= beta
        v = rmatvec(u)
        alpha = cublas.nrm2(v)
        alpha_cpu = alpha.get().item()
    if alpha_cpu > 0:
        v /= alpha
    itn = 0
    zetabar = alpha_cpu * beta_cpu
    alphabar = alpha_cpu
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0
    h = v.copy()
    hbar = cp.zeros(n, dtype)
    betadd = beta_cpu
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0
    normA2 = alpha_cpu * alpha_cpu
    maxrbar = 0
    minrbar = 1e100
    normA = alpha_cpu
    condA = 1
    normx = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta_cpu
    normar = alpha_cpu * beta_cpu
    if normar == 0:
        return (x, istop, itn, normr, normar, normA, condA, normx)
    while itn < maxiter:
        itn = itn + 1
        av = matvec(v)
        axpy(((m + 255) // 256,), (256,), (u, av, np.float64(-alpha_cpu), np.int64(m)))
        del av
        beta = cublas.nrm2(u)
        beta_cpu = beta.get().item()
        if beta_cpu > 0:
            u /= beta
            atu = rmatvec(u)
            axpy(
                ((n + 255) // 256,),
                (256,),
                (v, atu, np.float64(-beta_cpu), np.int64(n)),
            )
            del atu
            alpha = cublas.nrm2(v)
            alpha_cpu = alpha.get().item()
            if alpha_cpu > 0:
                v /= alpha
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta_cpu)
        thetanew = s * alpha_cpu
        alphabar = c * alpha_cpu
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        update(
            ((n + 255) // 256,),
            (256,),
            (
                hbar,
                h,
                x,
                v,
                np.float64(-(thetabar * rho / (rhoold * rhobarold))),
                np.float64(zeta / (rho * rhobar)),
                np.float64(-(thetanew / rho)),
                np.int64(n),
            ),
        )
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = np.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta_cpu * beta_cpu
        normA = np.sqrt(normA2)
        normA2 = normA2 + alpha_cpu * alpha_cpu
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = cublas.nrm2(x)
        normx = normx.get().item()
        test1 = normr / normb
        if normA * normr != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = np.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop > 0:
            break
    # The specialized input and all work vectors are already float64.
    return (x, istop, itn, normr, normar, normA, condA, normx)
