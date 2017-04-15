function varargout = guauViewer(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @guauViewer_OpeningFcn, ...
                   'gui_OutputFcn',  @guauViewer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

end


function guauViewer_OpeningFcn(hObject, eventdata, handles, varargin)

global path_to_images path_to_labels class_dict plot_colours ...
    conf_thresh plot_bool class_array slide_bool im_size

handles.output = hObject;

im_size = [375, 1242];
path_to_images = 'C:\Users\Donal\Dropbox\KITTI\data\testing\image\';
path_to_labels = 'C:\Users\Donal\Dropbox\KITTI\output\predictions\';
conf_thresh = 0.0;
slide_bool = 0;

% Create dictionary for plotting
classes = {'Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc','DontCare'};
class_num = num2cell(1:size(classes,2));
class_dict = containers.Map(classes,class_num);
plot_colours = ['y','m','c','r','g','b','w','k','w','k'];

% Populate pop-up menu
plot_bool = 'All';
class_array = [plot_bool classes];
set(handles.classPopUp,'String',class_array);
 
% Display first image and index
dirInit(handles)

guidata(hObject, handles);

end


function varargout = guauViewer_OutputFcn(hObject, eventdata, handles) 

varargout{1} = handles.output;

end


function goToTxt_Callback(hObject, eventdata, handles)

global go_to_idx

content = cellstr(get(hObject,'String'));
go_to_idx = round(str2double(content));

end

function goToTxt_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function curImTxt_Callback(hObject, eventdata, handles)


end


function curImTxt_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function previousButton_Callback(hObject, eventdata, handles)

global im_idx no_img

if im_idx == 0
    im_idx = no_img-1;
else
    im_idx = im_idx-1;
end

plotImage(handles)

end


function nextButton_Callback(hObject, eventdata, handles)

global im_idx no_img

if im_idx == no_img-1
    im_idx = 0;
else
    im_idx = im_idx+1;
end

plotImage(handles)

end


function confThres_Callback(hObject, eventdata, handles)

global conf_thresh

content = cellstr(get(hObject,'String'));
thres_temp = str2double(content);
if thres_temp >= 0 || thres_temp <= 1
    conf_thresh = thres_temp;
else
    set(handles.confThresTxt, 'String', num2str(conf_thresh))
end
plotImage(handles)

end


function confThres_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function listbox1_Callback(hObject, eventdata, handles)

end


function listbox1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function goToButton_Callback(hObject, eventdata, handles)

global im_idx go_to_idx no_img

if go_to_idx > no_img-1 || go_to_idx < 0
    go_to_idx = im_idx;
    set(handles.goToTxt, 'String', num2str(im_idx)) 
else
    im_idx = go_to_idx;
    plotImage(handles)
end

end


function plotImage(handles)

global path_to_images path_to_labels im_idx no_img class_dict plot_colours ...
    conf_thresh plot_bool im_size

set(handles.curImTxt, 'String', [num2str(im_idx) '/' num2str(no_img-1)])
im = imread(sprintf('%s/%06d.png',path_to_images,im_idx));
im = imresize(im,im_size);
axes(handles.ImagePane)
image(im)
axis off
axis image

objects = readLabelsGuau(path_to_labels,im_idx);

for obj_idx=1:numel(objects)
    object = objects(obj_idx);
    if object.conf >= conf_thresh && (strcmp(plot_bool,object.type) || strcmp(plot_bool,'All'))
        drawBox2DGuau(handles.ImagePane,objects(obj_idx),class_dict,plot_colours)
    elseif isnan(object.conf)
        drawBox2DGuau(handles.ImagePane,objects(obj_idx),class_dict,plot_colours)
    end
end
    
end

function plotImageTimer(handles)

global path_to_images path_to_labels im_idx no_img class_dict plot_colours ...
    conf_thresh plot_bool

im_idx = im_idx+1;
set(handles.curImTxt, 'String', [num2str(im_idx) '/' num2str(no_img-1)])
im = imread(sprintf('%s/%06d.png',path_to_images,im_idx));
axes(handles.ImagePane)
image(im)
axis off
axis image

objects = readLabelsGuau(path_to_labels,im_idx);

for obj_idx=1:numel(objects)
    object = objects(obj_idx);
    if object.conf >= conf_thresh && (strcmp(plot_bool,object.type) || strcmp(plot_bool,'All'))
        drawBox2DGuau(handles.ImagePane,objects(obj_idx),class_dict,plot_colours)
    end
end
    
end


function classPopUp_Callback(hObject, eventdata, handles)

global plot_bool class_array

plot_bool = class_array(get(handles.classPopUp,'Value'));
plotImage(handles)

end


function classPopUp_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

end


function slideButton_Callback(hObject, eventdata, handles)

% TO DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sec_per_im = 1.5;
t = timer('Name',slide_show,'Period',sec_per_im);
t.timerFcn = @(handles)plotImageTimer;
start(t)
delete(t)

end


function randButton_Callback(hObject, eventdata, handles)

global no_img im_idx

im_idx = randi([0 no_img-1]);
plotImage(handles)

end


function imDirBtn_Callback(hObject, eventdata, handles)

global path_to_images

p_t_i = path_to_images;
path_to_images = uigetdir('','Set image directory');
if sum(path_to_images == 0)
    disp('Image directory unchaged!!!')
    path_to_images = p_t_i;
end
dirInit(handles)

end


function dirInit(handles)

global path_to_images no_img im_idx
im_idx = 0;
D = dir([path_to_images, '\*.png']);
no_img = length(D(not([D.isdir])));
plotImage(handles)
set(handles.curImTxt, 'String', ['0/' num2str(no_img-1)])

end


function lblDirBtn_Callback(hObject, eventdata, handles)

global path_to_labels

p_t_l = path_to_labels;
path_to_labels = uigetdir('','Set label directory');
if sum(path_to_labels == 0)
    disp('Label directory unchaged!!!')
    path_to_labels = p_t_l;
end
dirInit(handles)

end