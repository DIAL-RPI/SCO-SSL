import vtk
from vtk.util.numpy_support import vtk_to_numpy
import math
import sys
from jet_color_table import jet_colormap

def get_cube_axes(bounds, ren):
    cubeAxesActor = vtk.vtkCubeAxesActor()
    cubeAxesActor.SetBounds(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
    cubeAxesActor.SetCamera(ren.GetActiveCamera())
    cubeAxesActor.GetTitleTextProperty(0).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(0).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.GetTitleTextProperty(1).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(1).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 0.0)
    cubeAxesActor.SetXAxisRange(-1, +1)
    cubeAxesActor.SetYAxisRange(-1, +1)
    cubeAxesActor.SetZAxisRange(-1, +1)
    cubeAxesActor.SetScreenSize(12)
    cubeAxesActor.SetLabelOffset(5)
    cubeAxesActor.SetVisibility(True)
    cubeAxesActor.SetFlyMode(0)
    cubeAxesActor.GetXAxesLinesProperty().SetLineWidth(2)
    cubeAxesActor.GetXAxesLinesProperty().SetColor(0,0,0)
    cubeAxesActor.GetYAxesLinesProperty().SetLineWidth(2)
    cubeAxesActor.GetYAxesLinesProperty().SetColor(0,0,0)
    cubeAxesActor.GetZAxesLinesProperty().SetLineWidth(2)
    cubeAxesActor.GetZAxesLinesProperty().SetColor(0,0,0)
    cubeAxesActor.GetXAxesGridlinesProperty().SetLineWidth(0.5)
    cubeAxesActor.GetXAxesGridlinesProperty().SetColor(0.5, 0.5, 0.5)
    cubeAxesActor.GetYAxesGridlinesProperty().SetLineWidth(0.5)
    cubeAxesActor.GetYAxesGridlinesProperty().SetColor(0.5, 0.5, 0.5)
    cubeAxesActor.GetZAxesGridlinesProperty().SetLineWidth(0.5)
    cubeAxesActor.GetZAxesGridlinesProperty().SetColor(0.5, 0.5, 0.5)
    cubeAxesActor.DrawXGridlinesOn()
    cubeAxesActor.DrawYGridlinesOn()
    cubeAxesActor.DrawZGridlinesOn()
    cubeAxesActor.SetDrawXInnerGridlines(False)
    cubeAxesActor.SetDrawYInnerGridlines(False)
    cubeAxesActor.SetDrawZInnerGridlines(False)
    cubeAxesActor.XAxisMinorTickVisibilityOff()
    cubeAxesActor.YAxisMinorTickVisibilityOff()
    cubeAxesActor.ZAxisMinorTickVisibilityOff()
    cubeAxesActor.SetXAxisVisibility(True)
    cubeAxesActor.SetYAxisVisibility(True)
    cubeAxesActor.SetZAxisVisibility(True)
    cubeAxesActor.SetXAxisTickVisibility(True)
    cubeAxesActor.SetYAxisTickVisibility(True)
    cubeAxesActor.SetZAxisTickVisibility(True)
    cubeAxesActor.SetXTitle('')
    cubeAxesActor.SetYTitle('')
    cubeAxesActor.SetZTitle('')
    cubeAxesActor.SetGridLineLocation(2)
    cubeAxesActor.SetTickLocation(1)

    return cubeAxesActor

def extract_roi(reader, roi_label, color, opacity=1, represent='surface'):
    # convert the segmentation to binary mask by thresholding
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdBetween(roi_label, roi_label)
    threshold.SetInValue(1.0)
    threshold.SetOutValue(0.0)

    # apply marching cube algorithm to convert pixel volume to mesh grid
    iso=vtk.vtkMarchingCubes()
    iso.SetInputConnection(threshold.GetOutputPort())
    iso.SetValue(0, 0.5)
    iso.ComputeGradientsOn()
    iso.ComputeNormalsOff()
    iso.ComputeScalarsOff()

    # apply smoothing filter on the binary mask to get a better visualization
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(iso.GetOutputPort())
    smooth.SetNumberOfIterations(20)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.SetFeatureAngle(120)
    smooth.SetPassBand(0.01)
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()

    # calculate center of mass of the binary mask
    cm = vtk.vtkCenterOfMass()
    cm.SetInputConnection(smooth.GetOutputPort())
    cm.SetUseScalarsAsWeights(False)
    cm.Update()
    roi_center = cm.GetCenter()

    # generate vtk actor of the segmented volume and its outline box
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(smooth.GetOutputPort())

    roi_actor = vtk.vtkActor()
    roi_actor.SetMapper(cube_mapper)
    roi_actor.GetProperty().SetColor(color[0],color[1],color[2])
    roi_actor.GetProperty().SetOpacity(opacity)
    if represent == 'point':
        roi_actor.GetProperty().SetRepresentationToPoints()
    elif represent == 'wire':
        roi_actor.GetProperty().SetRepresentationToWireframe()
    elif represent == 'surface':
        roi_actor.GetProperty().SetRepresentationToSurface()
    roi_actor.GetProperty().SetLineWidth(1)

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(smooth.GetOutputPort())
    outline_mapper = vtk.vtkPolyDataMapper()
    outline_mapper.SetInputConnection(outline.GetOutputPort())
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(outline_mapper)
    outline_actor.GetProperty().SetColor(0.8,0.8,0.8)
    outline_actor.GetProperty().SetOpacity(0.9)
    outline_actor.GetProperty().SetRepresentationToWireframe()
    outline_actor.GetProperty().SetLineWidth(2)

    return roi_actor, roi_center, outline_actor

def extract_distancemap(pd_reader, gt_reader, pd_label, gt_label, color, opacity=1, represent='surface', max_scalar=None):
    # convert the predicted segmentation to binary mask by thresholding
    threshold1 = vtk.vtkImageThreshold()
    threshold1.SetInputConnection(pd_reader.GetOutputPort())
    threshold1.ThresholdBetween(pd_label, pd_label)
    threshold1.SetInValue(1.0)
    threshold1.SetOutValue(0.0)

    # apply marching cube algorithm to convert pixel volume to mesh grid
    iso1=vtk.vtkMarchingCubes()
    iso1.SetInputConnection(threshold1.GetOutputPort())
    iso1.SetValue(0, 0.5)
    iso1.ComputeGradientsOn()
    iso1.ComputeNormalsOff()
    iso1.ComputeScalarsOff()

    # keep the largest connected component in the binary mask
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputConnection(iso1.GetOutputPort())
    connect.SetExtractionModeToLargestRegion()

    # apply smoothing filter on the binary mask to get a better visualization
    smooth1 = vtk.vtkWindowedSincPolyDataFilter()
    smooth1.SetInputConnection(connect.GetOutputPort())
    smooth1.SetNumberOfIterations(20)
    smooth1.BoundarySmoothingOff()
    smooth1.FeatureEdgeSmoothingOff()
    smooth1.SetFeatureAngle(120)
    smooth1.SetPassBand(0.02)
    smooth1.NonManifoldSmoothingOn()
    smooth1.NormalizeCoordinatesOn()

    # convert the ground-truth segmentation to binary mask by thresholding
    threshold2 = vtk.vtkImageThreshold()
    threshold2.SetInputConnection(gt_reader.GetOutputPort())
    threshold2.ThresholdBetween(gt_label, gt_label)
    threshold2.SetInValue(1.0)
    threshold2.SetOutValue(0.0)

    # apply marching cube algorithm to convert pixel volume to mesh grid
    iso2=vtk.vtkMarchingCubes()
    iso2.SetInputConnection(threshold2.GetOutputPort())
    iso2.SetValue(0, 0.5)
    iso2.ComputeGradientsOn()
    iso2.ComputeNormalsOff()
    iso2.ComputeScalarsOff()

    # apply smoothing filter on the binary mask to get a better visualization
    smooth2 = vtk.vtkWindowedSincPolyDataFilter()
    smooth2.SetInputConnection(iso2.GetOutputPort())
    smooth2.SetNumberOfIterations(20)
    smooth2.BoundarySmoothingOff()
    smooth2.FeatureEdgeSmoothingOff()
    smooth2.SetFeatureAngle(120)
    smooth2.SetPassBand(0.02)
    smooth2.NonManifoldSmoothingOn()
    smooth2.NormalizeCoordinatesOn()

    # calculate surface distance between predicted segmentation and ground-truth segmentation
    dist = vtk.vtkDistancePolyDataFilter()
    dist.SetInputConnection(0, smooth1.GetOutputPort())
    dist.SetInputConnection(1, smooth2.GetOutputPort())
    dist.SignedDistanceOff()
    dist.Update()
    
    dist_arr = vtk_to_numpy(dist.GetOutput().GetPointData().GetScalars())

    # generate vtk actor for the "distance-rendered" volume
    lut = vtk.vtkLookupTable()
    jet = jet_colormap()
    lut.SetNumberOfColors(len(jet))
    for i in range(len(jet)):
        lut.SetTableValue(i, jet[i][0], jet[i][1], jet[i][2], 1.0)

    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(dist.GetOutputPort())
    cube_mapper.SetLookupTable(lut)
    if max_scalar is None:
        cube_mapper.SetScalarRange(0, dist_arr.max())
    else:
        cube_mapper.SetScalarRange(0, max_scalar)

    roi_actor = vtk.vtkActor()
    roi_actor.SetMapper(cube_mapper)
    roi_actor.GetProperty().SetColor(color[0],color[1],color[2])
    roi_actor.GetProperty().SetOpacity(opacity)
    if represent == 'point':
        roi_actor.GetProperty().SetRepresentationToPoints()
    elif represent == 'wire':
        roi_actor.GetProperty().SetRepresentationToWireframe()
    elif represent == 'surface':
        roi_actor.GetProperty().SetRepresentationToSurface()
    roi_actor.GetProperty().SetLineWidth(1)

    return roi_actor, dist_arr.min(), dist_arr.max()

gt_dir = '/home/username/data/label' # directory where the ground-truth segmentation masks stored
pd_dir = '/home/username/proj/result' # directory where the predicted segmentation masks stored
casename = 'Case0001' # name of the case you want to visualize
    
gt_filename = '{}/{}.nii.gz'.format(gt_dir, casename) # full filename of the ground-truth segmentation mask file (we assume the segmentation mask is named as its casename and stored in Nifti format)
pd_filename = '{}/{}.nii.gz'.format(pd_dir, casename) # full filename of the predicted segmentation mask file (we assume the segmentation mask is named as its casename and stored in Nifti format)

# read grount-truth segmentation
gt_reader = vtk.vtkNIFTIImageReader()
gt_reader.SetFileName(gt_filename)
gt_reader.Update()

# generate vtk actor of the ground-truth volume
gt_actor, gt_center, outline_actor = extract_roi(gt_reader, roi_label=1, color=(0.1,0.1,1.0), opacity=1.0, represent='surface')
gt_boundrs = gt_actor.GetBounds()
gt_height = gt_boundrs[5] - gt_boundrs[4]
bounds = gt_actor.GetBounds()

# calculate camera position using the specified distance 'dist' and angles: 'a' and 'b'
dist = gt_height * 4
a = 7 * math.pi / 180
b = 120 * math.pi / 180
camera_pos = [
    gt_center[0] - dist * math.cos(a) * math.cos(b),
    gt_center[1] - dist * math.cos(a) * math.sin(b),
    gt_center[2] + dist * math.sin(a)
]

# set light source positions according to the center of the ground-truth volume
light_pos = []
light_pos.append([gt_center[0],gt_center[1],gt_center[2]-dist])
light_pos.append([gt_center[0],gt_center[1],gt_center[2]+dist])
light_pos.append([gt_center[0],gt_center[1]+dist,gt_center[2]])
light_pos.append([gt_center[0],gt_center[1]-dist,gt_center[2]])
light_pos.append([gt_center[0]+dist,gt_center[1],gt_center[2]])
light_pos.append([gt_center[0]-dist,gt_center[1],gt_center[2]])

# read predicted segmentation
pd_reader = vtk.vtkNIFTIImageReader()
pd_reader.SetFileName(pd_filename)
pd_reader.Update()

# generate vtk actor of the distance volume
dist_actor, _, max_dist = extract_distancemap(pd_reader, gt_reader, pd_label=1, gt_label=1, color=(1,0.5,0), opacity=1.0, represent='surface', max_scalar=None)
dist_actor.GetMapper().SetScalarRange(0, max_dist)

ren = vtk.vtkRenderer()
ren.AddActor(outline_actor)
ren.AddActor(dist_actor)

# generate vtk actor of the axes and grid
cubeAxesActor = get_cube_axes(bounds, ren)
ren.AddActor(cubeAxesActor)    

camera = vtk.vtkCamera()
camera.SetFocalPoint(gt_center)
camera.SetPosition(camera_pos)
camera.ComputeViewPlaneNormal() 
camera.SetViewUp([0,0,1])
ren.SetActiveCamera(camera)

ren.RemoveAllLights()
for i in range(len(light_pos)):
    light = vtk.vtkLight()
    light.SetColor(1,1,1)
    light.SetIntensity(0.6)
    light.SetPosition(light_pos[i])
    light.SetFocalPoint(ren.GetActiveCamera().GetFocalPoint())
    ren.AddLight(light)

ren_win = vtk.vtkRenderWindow()
ren.SetBackground(1.0, 1.0, 1.0)
ren_win.AddRenderer(ren)
ren_win.SetSize(1000,1000)
ren_wit = vtk.vtkRenderWindowInteractor()
ren_wit.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
ren_wit.SetRenderWindow(ren_win)
ren_win.Render()

ren_wit.Start()