import sys

args = sys.argv
if not(2 <= len(args) <= 3):
   print("Invalid arguments: mapper.py needs 1 or 2 arguments")
   print("mapper.py <shape> [offset]")
   print("Example: python mapper.py 32x12x32 0,4,0")
   sys.exit(1)


dims = [int(dim) for dim in  args[1].split("x")]
if len(args)==3:
    offs = [int(dim) for dim in  args[2].split(",")]
else:
    offs = [0, 0, 0]

mx = dims[0]
my = dims[1]
mz = dims[2]
if mx % 2 != 0:
   print("X dim must be 4*m (m:arbitrary)")
   sys.exit(1)
#if my % 3 != 0:
#   print("Y dim must be 3*m (m:arbitrary)")
#   sys.exit(1)
if mz % 8 != 0:
   print("Z dim must be 8*m (m:arbitrary)")
   sys.exit(1)

jx = [[i for i in range(2)] for j in range(mx//2)]
#jy = [[i for i in range(3)] for j in range(my//3)]
jz = [[i for i in range(2)] for j in range(mz//2)]
nx = mx//2
ny = my//3
nz = mz//2

for ia in list(range(2)):
   for ix in list(range(nx)):
      jx[ix][ia] = 2*ia*nx - 2*ia*ix - ia+ix

for ic in list(range(2)):
   for iz in list(range(nz)):
      jz[iz][ic] = 2*ic*nz - 2*ic*iz - ic+iz
"""
iy=0
ib=0
id=1
ie=2

for ia in range(ny*3):
   jy[iy][ib] = ia
   if ia % 3 == 2:
      ib = (ib+ie) % 3
   else:
      iy = iy + id
   if iy == ny:
      id = -1
      ie = 1
      iy = ny - 1
      if ny % 2 == 0:
         ib = (ib+1) % 3
      else:
         ib = (ib+2) % 3
   else:
      if iy == -1:
         id = 0
         iy = 0
         ib = (ib+2) % 3
"""
iboba=[0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0]
ibobc=[0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0]
ibobz=[0,1,1,0,0,0,1,1,2,2,3,3,3,2,2,3]
for ky in range(my):
   for ix in range(nx):
      for iz in range(nz//4):
         for ii in range(16):
            kx = jx[ix][iboba[ii]]
            kz = jz[iz*4+ibobz[ii]][ibobc[ii]]
            print("("+str(kx+offs[0])+","+str(ky+offs[1])+","+str(kz+offs[2])+")")

exit(0)
