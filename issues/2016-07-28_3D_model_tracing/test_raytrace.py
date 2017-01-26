from rockfish.tomography.forward import raytrace_from_ascii


vmfile = '11.11.vm'

# simple geometry and picks
instfile = 'inst.dat' 
shotfile = 'shot.dat'
pickfile = 'pick.dat'

f = open(instfile, 'w')
f.write('100 25. 0.0 10.0\n')
f.write('101 30. 0.0 20.0\n')
f.close()

f = open(shotfile, 'w')
f.write('9000 5. 0.0 0.006\n')
f.write('9001 10. 0.0 0.006\n')
f.close()

f = open(pickfile, 'w')
f.write('100 9000 1 0 9.999 9.999 0.000\n')
f.write('101 9001 1 0 9.999 9.999 0.000\n')
f.close()


rayfile = 'test.rays'

raytrace_from_ascii(vmfile, rayfile, instfile=instfile,
                    shotfile=shotfile, pickfile=pickfile, verbose=1000)
