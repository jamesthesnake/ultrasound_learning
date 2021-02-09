load ../data/test_PICMUS.mat img img_x img_z

img_idx = 1;
% Make complex
img = img(1,:,:,:,:) + 1i * img(2,:,:,:,:);
img = img(:,:,:,:,img_idx);

% Make B-mode image
bimg = squeeze(abs(sum(img, 2)));
bimg = bimg / max(bimg(:));

% Load predicted image
load ../runs/pretrained/results_PICMUS.mat
pimg = squeeze(p(:,:,1,img_idx));
% Minimize mean squared error between pimg, bimg
pimg = minimize_msq_bmode(pimg, bimg);

figure(1); clf
set(gcf,'Position',[100 100 1000 600],'PaperPositionMode','auto')
subplot(121)
imagesc(img_x*1e3, img_z*1e3, db(bimg), [-40 0]);
axis image; colormap gray;
xlabel('Azimuth (mm)', 'FontSize', 14)
ylabel('Depth (mm)', 'FontSize', 14)
title('DAS B-mode', 'FontSize', 16)

subplot(122)
imagesc(img_x*1e3, img_z*1e3, db(pimg), [-40 0]);
axis image; colormap gray;
xlabel('Azimuth (mm)', 'FontSize', 14)
ylabel('Depth (mm)', 'FontSize', 14)
title('NN B-mode', 'FontSize', 16)


function [img_opt, w_opt] = minimize_msq_bmode(img, ref)
% Minimize the mean squared error between the ref and img by multiplying
% img by some positive scalar

img_opt = 0*img;
[~,~,ni] = size(img);
w_opt = zeros(ni, 1, 'single');

for i = 1:ni
	I = reshape(img(:,:,i),[],1);
	R = reshape(ref(:,:,i),[],1);
	w_opt(i) = abs(sum(I.*R) / sum(I.*I));
	img_opt(:,:,i) = img(:,:,i) * w_opt(i);
end
end