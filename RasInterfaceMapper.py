#/usr/bin/python
import sys
import mdtraj as md
from numpy import *
from numpy.linalg import inv
from glob import glob
from math import cos, sin, atan2

#Last update 8/23/2013 David Walker
#TODO: Make sure that the angle calcuor works, like at all.
#Notice you should fix periodic boundary conditions before running this program


#We will need some simple math functions to describe the geometry of the nitrile
def find_vector((x,y,z),(i,j,k)):
    return ((i-x),(j-y),(k-z))

def midpoint((x,y,z), (i,j,k)):
    vec = ((i-x),(j-y),(k-z))
    return ((x +vec[0]*0.5),(y+vec[1]*0.5),(z+vec[2]*0.5))

def norm(vector):
    #return sqrt(dot(vector, vector))
    m_vec=mag(vector)
    if m_vec==0.0:
            return vector - vector
    else:
            return vector/m_vec

def unit_vector(vector):
    return vector/norm(vector)

def angle_between(v1, v2):
    v1_u = norm(v1)
    v2_u = norm(v2)
    angle = arccos(dot(v1_u , v2_u))
    if isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return pi
    return angle

def dist( a1, a2) :
    a = a1 - a2
    return (dot(a,a))**0.5

def mag (vector):
    return dist( vector, array((0,0,0)))

mutant = sys.argv[1] #This is just the name of the mutant for which the gromacs file naming schemes are based on. This is specific to my ("David Walker") work.
xtcfiles = glob('align*%s*.xtc' % mutant)
print 'found %d xtc files for %s' % (len(xtcfiles), mutant)
RasPlane = range(28, 44) #This is the residue number of each residue I24-S39. mdtraj starts numbering at 0 and not 1. Also there are the 5 residues added to the PDB that aren't in the numbering
#It should be noted that this is specific to how I (David Walker) built my models. You may have to use this differently.

#Lets look through all the xtc files in an iteration
for xtc in xtcfiles:
    bins=zeros((15,15))
    PlaneList=[]
    PlaneTop=[] #This will be our plane topology list, because the plane might move during the simulation.
    pdb = xtc.replace('xtc', 'pdb') # The topology must be supplied by a PDB file.
    writefile = open('RasPlane_'+xtc.replace('xtc', 'xvg'), "w")
    except_count = 0
    try:
        traj = md.load(xtc, top = pdb) #Loading up the XTC file.
        print "Reading %s using the file %s as a topology." % (xtc, pdb)

        
        #Let us find the nitrile for our experiment.
        cnc_residue = [r.index for r in traj.topology.residues if r.name == 'CNC']
        CNC = cnc_residue[0]
        cnc_cd = [a.index for a in traj.topology.residue(CNC).atoms if a.name == "CD"] #Get the delta carbon on the nitrile.
        cnc_ne = [a.index for a in traj.topology.residue(CNC).atoms if a.name == "NE"] #Get the epsilon nitrile on the nitrile.
        
        for t in range(traj.n_frames): # iterate over all the frames.    
            ras_xy = ones((16,3)) # Initialize the matrix as a bunch of ones. This will fill up with CA positions for the X and Y coordinates for residues Ras I24-S39
            tras_z = zeros(16) # This will fill up with the CA positions for the Z coordinates for RasI24-S39. This is really tranposed from the original shape we want.
            
            for i in RasPlane: #Lets build our Ras plane
                surf_atoms = [a.index for a in traj.topology.residue(i).atoms if a.name == "CA"] # Pull only the CA atoms
                ras_xyz = traj.xyz[t][surf_atoms]
                ras_xy[i-28][0]=ras_xyz[0][0]
                ras_xy[i-28][1]=ras_xyz[0][1]
                tras_z[i-28]=ras_xyz[0][2]
 
            ras_z = tras_z.transpose()
            B1 = dot(ras_xy.transpose(), ras_xy)
            B2 = dot(ras_xy.transpose(), ras_z)

            try:
                RasInterface = dot(inv(B1), B2)
                x_min = min(ras_xy.transpose()[0])
                x_max = max(ras_xy.transpose()[0])
                y_min = min(ras_xy.transpose()[1])
                y_max = max(ras_xy.transpose()[1])
            except: 
                print "\n\tThere was an error in the dot product. We are just using the last known value as an approximation. We will keep the last frames values\n"
                RasInterface = PlaneList[t-1]
            #print "The plane is given by z = %1.3f(x) + %1.3f(y) + %1.3f" %(RasInterface[0], RasInterface[1], RasInterface[2])
            PlaneList.append(RasInterface)

            dx = (x_max - x_min)/15
            dy = (y_max - y_min)/15

            z_min = RasInterface[0]*x_min+RasInterface[1]*y_min+RasInterface[2]
            z_max = RasInterface[0]*x_max+RasInterface[1]*y_max+RasInterface[2]
            
            plane_vec = ((x_max - x_min),(y_max-y_min), (z_max - z_min))
            ras_normal = ((float(RasInterface[0]), float(RasInterface[1]), -1))
            N = -norm(ras_normal)

            #Lets Build an all CA coordinate plane
            CA_atoms = [a.index for a in traj.topology.atoms if a.name == "CA"]
            CA_xy = ones((len(CA_atoms), 3))
            tCA_z = zeros(len(CA_atoms))
            CA_xyz = traj.xyz[t][CA_atoms]
            for i in range(len(CA_atoms)):
                CA_xy[i][0] = CA_xyz[i][0]
                CA_xy[i][1] = CA_xyz[i][1]
                tCA_z[i] = CA_xyz[i][2]

            CA_z = tCA_z.transpose()
            M1 = dot(CA_xy.transpose(), CA_xy)
            M2 = dot(CA_xy.transpose(), CA_z)
            CA_plane = dot(inv(M1), M2)
            #print CA_xy 
            CA_normal = ((float(CA_plane[0]), float(CA_plane[1]), -1))

            #Determine the line orthoganal to both the Interface and CA_plane
            cx = cross(ras_normal, CA_normal) #Form a vector normal to the CA plane and Ras Plane

            A = matrix(((ras_normal[1], -1), (CA_normal[1], -1)))
            B = matrix((-1*ras_normal[2], -1*CA_normal[2])).T
            result = dot(inv(A),B)

            #Redefining the surface plane in term of the vectors which it spans
            PV1 = norm(cx)
            PV = cross(cx, N)
            PV2 = norm(N)
            p1 = ((1, 0, float(ras_normal[0]-1)))
            p2 =((2,0,float(ras_normal[0]*2-1)))
            xaxis = norm(array(p2)-array(p1))
            if PV1[1] >= 0:
                ref_angle = arccos(dot(PV1,xaxis))*180/pi
            elif PV[1] < 0:
                ref_angle = -1 * arccos(dot(PV1,xaxis))*180/pi

            #Lets get the information on the nitrile for each frame.
            cd_coords = traj.xyz[t][cnc_cd]
            ne_coords = traj.xyz[t][cnc_ne]
            cnc_vec = find_vector(cd_coords[0], ne_coords[0])
            cnc_mid = midpoint(cd_coords[0], ne_coords[0]) #nitrile midpoint XYZ coordinates.
            cnc_dist = float(sqrt(((RasInterface[0]*cnc_mid[0]+RasInterface[1]*cnc_mid[1]+RasInterface[2])-cnc_mid[2])**2))

            #assign the nitrile to the topology bins
            x_bin = round((cnc_mid[0] - x_min)/dx)
            y_bin = round((cnc_mid[1] - y_min)/dy)
            print "%0.5f, %0.5f" % (x_bin, y_bin)
            
            #Let us try and find the angle of elevation with the Ras Plane
            cnc_angle = angle_between(plane_vec, cnc_vec)*180/pi
            #print cnc_angle*180/pi

            #Let us try and find the azimuthal angle now
            cnc_norm = norm(cnc_vec)
            #print cnc_norm
            #print PV1
            proj_cnc = norm(dot(PV1, cnc_norm)/dot(PV1,PV1)*PV1+dot(PV2,cnc_norm)/dot(PV2,PV2)*PV2)
            if proj_cnc[1]>= 0:
                CN_angle = arccos(dot(xaxis, proj_cnc)/(mag(xaxis)*mag(proj_cnc)))*180/pi

            elif proj_cnc[1] < 0:
                CN_angle = -1*arccos(dot(xaxis, proj_cnc)/(mag(xaxis)*mag(proj_cnc)))*180/pi

            azim = (CN_angle - ref_angle) - 90

            output = '%d %d %0.5f %0.5f %0.5f %0.5f %0.5f %0.5f\n' % (x_bin, y_bin, cnc_angle, azim, cnc_dist, cnc_vec[0], cnc_vec[1], cnc_vec[2])
            writefile.write(output)
        writefile.close()

        #By now whe should have all the information for how the binding plane changes and how the nitrile moves.
        
    except ValueError:
        print 'No such file as %s and/or %s' % (xtc, pdb)

print 'You have all of the information about the nitrile for each snapshot in simulation. You should double check to make sure that you do not have any issues with periodic boundary conditions'
