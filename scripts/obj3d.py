
#%pip install PyOpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

import numpy as np



class objFile:

    #-------------------------------------------------------------------------------
    #                                __init__
    #-------------------------------------------------------------------------------
    def __init__(self, path):

        if path is not None:
            self.load_obj(path)

        return


    #-------------------------------------------------------------------------------
    #                                load_obj
    #-------------------------------------------------------------------------------
    def load_obj(self, path):

        # Open the file
        file = open(path, 'r')

        # Create lists to contain verticies, normals, faces
        vertices = []
        normals = []
        faces = []

        # Read the contents of the file, one line at a time
        for line in file:
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            
            values = line.split(' ')

            lineType = values[0].strip().lower()

            if lineType == 'v':
                v = list(map(float, values[1:]))
                vertices.append(v)

            elif lineType == 'vn':
                vn = list(map(float, values[1:]))
                normals.append(vn)
            
            elif lineType == 'f':
                f = list(map(int, values[1:]))
                faces.append(f)

        # Close the file
        file.close()

        self.vertices = vertices
        self.normals = normals
        self.faces = faces

        return


    #-------------------------------------------------------------------------------
    #                                  render
    #-------------------------------------------------------------------------------
    def render(self, justShowPoints=False):

        glColor3f(1.0, 1.0, 1.0)

        glFrontFace(GL_CCW)

        for face in self.faces:

            # bunny object only has vertices
            vertexIndeces = face

            # Get the points (coordinates) for each vertex
            try:
                pt1 = self.vertices[vertexIndeces[0]-1]
                pt2 = self.vertices[vertexIndeces[1]-1]
                pt3 = self.vertices[vertexIndeces[2]-1]

            except:
                print(vertexIndeces)
                print(len(self.vertices))
                raise Exception('error. invalid index. (render method)')

            
            # TODO:  Compute the normals one time, then save them to the list...

            pt1m = np.matrix(pt1)
            pt2m = np.matrix(pt2)
            pt3m = np.matrix(pt3)

            # Create two edge vectors
            U = pt2m - pt1m
            V = pt3m - pt1m

            # Compute the normal
            N = np.cross(U, V)

            # Make the normal a unit vector (length 1)
            N /= np.linalg.norm(N)

            if justShowPoints:

                glPointSize(1.0)

                glBegin(GL_POINTS)
                glVertex3f(pt1[0], pt1[1], pt1[2])
                glVertex3f(pt2[0], pt2[1], pt2[2])
                glVertex3f(pt3[0], pt3[1], pt3[2])
                glEnd()

            else:
                glNormal3f(N[0,0], N[0, 1], N[0, 2])

                glBegin(GL_TRIANGLES)
                glVertex3f(pt1[0], pt1[1], pt1[2])
                glVertex3f(pt2[0], pt2[1], pt2[2])
                glVertex3f(pt3[0], pt3[1], pt3[2])
                glEnd()
            

            
            
            
        return

# End of File
