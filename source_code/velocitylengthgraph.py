from jelly import Cube

cube = Cube()

cube.saveDroptestGraph(3000, "cube_normal_3000")

center = cube.getCenter()
cube.translate(-1 * center)
cube.randRotate()
cube.translate(center)

cube.saveDroptestGraph(6000, "cube_rotated_6000")