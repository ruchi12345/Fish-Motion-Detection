%
% Copyright (c) 2013, Research Infinite Solutions llp (http://www.researchinfinitesolutions.com/)
% 
%
% 
% Project Title: Stock Market prediction using ANFIS
% Publisher:  (http://www.researchinfinitesolutions.com/)
% 
% Developer: Ruchi Mehra (Member of Research Infinite Solutions llp)
% 
% Contact Info: info@researchinfinitesolutions.com, ruchi@webtunix.com

% For training in Human Vision and Data Science contact at  ::::
% ruchi@webtunix.com
%% 




clear
clc
close all



% Object for reading video file.



filename = '10p5_3.avi';
hVidReader = vision.VideoFileReader(filename, 'ImageColorSpace', 'RGB',...
                              'VideoOutputDataType', 'single');

hOpticalFlow = vision.OpticalFlow('OutputValue', 'Horizontal and vertical components in complex form', 'ReferenceFrameDelay', 3);

hMean1 = vision.Mean;

hMean2 = vision.Mean('RunningMean', true);

hMedianFilt = vision.MedianFilter;

hclose = vision.MorphologicalClose('Neighborhood', strel('line',5,45));




%Create a System object to detect foreground using gaussian mixture models.
hfdet = vision.ForegroundDetector(...
        'NumTrainingFrames', 20, ...     % only 5 because of short video
        'InitialVariance', (30/255)^2); % initial standard deviation of 30/255

    hblob = vision.BlobAnalysis(...
    'CentroidOutputPort', false, 'AreaOutputPort', true, ...
    'BoundingBoxOutputPort', true, 'OutputDataType', 'double', ...
    'MinimumBlobArea', 250, 'MaximumBlobArea', 3600, 'MaximumCount', 80);

herode = vision.MorphologicalErode('Neighborhood', strel('square',2));

hshapeins1 = vision.ShapeInserter('BorderColor', 'Custom', ...
                                  'CustomBorderColor', [1 0 0]);

                              hshapeins2 = vision.ShapeInserter( 'Shape','Lines', ...
                                   'BorderColor', 'Custom', ...
                                   'CustomBorderColor', [255 255 0]);
 
                               htextins = vision.TextInserter('Text', '%4d', 'Location',  [1 1], ...
                               'Color', [1 1 1], 'FontSize', 24);
                           sz = get(0,'ScreenSize');
% pos = [20 sz(4)-300 200 200];
pos = [200 sz(4)-900 800 800];
hVideo1 = vision.VideoPlayer('Name','Original Video','Position',pos);
pos(1) = pos(1)+220; % move the next viewer to the right
hVideo2 = vision.VideoPlayer('Name','Motion Vector','Position',pos);
pos(1) = pos(1)+220;
hVideo3 = vision.VideoPlayer('Name','Thresholded Video','Position',pos);
pos(1) = pos(1)+220;
hVideo4 = vision.VideoPlayer('Name','Results','Position',pos);

% Initialize variables used in plotting motion vectors.
lineRow   =  30;
firstTime = true;
motionVecGain  = 30;
borderOffset   = 5;
decimFactorRow = 5;
decimFactorCol = 5;
while ~isDone(hVidReader)  % Stop when end of file is reached
    frame  = step(hVidReader);  % Read input video frame
    grayFrame = rgb2gray(frame);
    ofVectors = step(hOpticalFlow, grayFrame);   % Estimate optical flow

    % The optical flow vectors are stored as complex numbers. Compute their
    % magnitude squared which will later be used for thresholding.
    y1 =ofVectors .* conj(ofVectors);
    % Compute the velocity threshold from the matrix of complex velocities.
    vel_th = 0.5 * step(hMean2, step(hMean1, y1));

    % Threshold the image and then filter it to remove speckle noise.
    segmentedObjects = step(hMedianFilt, y1 >= vel_th);

    % Thin-out the parts of the road and fill holes in the blobs.
    segmentedObjects = step(hclose, step(herode, segmentedObjects));

    % Estimate the area and bounding box of the blobs.
    [area, bbox] = step(hblob, segmentedObjects);
    % Select boxes inside ROI (below white line).
    Idx = bbox(:,1) > lineRow;

    % Based on blob sizes, filter out objects which can not be cars.
    % When the ratio between the area of the blob and the area of the
    % bounding box is above 0.4 (40%), classify it as a car.
    ratio = zeros(length(Idx), 1);
    ratio(Idx) = single(area(Idx,1))./single(bbox(Idx,3).*bbox(Idx,4));
    ratiob = ratio > 0.4;
    count = int32(sum(ratiob));    % Number of cars
    bbox(~ratiob, :) = int32(-1);

    % Draw bounding boxes around the tracked cars.
    y2 = step(hshapeins1, frame, bbox);

    % Display the number of cars tracked and a white line showing the ROI.
    y2(22:23,:,:)   = 1;   % The white line.
    y2(1:15,1:30,:) = 0;   % Background for displaying count
    result = step(htextins, y2, count);

    % Generate coordinates for plotting motion vectors.
    if firstTime
      [R C] = size(ofVectors);            % Height and width in pixels
      RV = borderOffset:decimFactorRow:(R-borderOffset);
      CV = borderOffset:decimFactorCol:(C-borderOffset);
      [Y X] = meshgrid(CV,RV);
      firstTime = false;
    end

    % Calculate and draw the motion vectors.
    tmp = ofVectors(RV,CV) .* motionVecGain;
    lines = [Y(:), X(:), Y(:) + real(tmp(:)), X(:) + imag(tmp(:))];
    motionVectors = step(hshapeins2, frame, lines);

    % Display the results
    step(hVideo1, frame);            % Original video
    step(hVideo2, motionVectors);    % Video with motion vectors
    step(hVideo3, segmentedObjects); % Thresholded video
    step(hVideo4, result);           % Video with bounding boxes
end

release(hVidReader);