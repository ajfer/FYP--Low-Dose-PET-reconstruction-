clc
close all
clear all

% nifti files for Low-dose, High-dose and MR 
low = mat2gray(niftiread('/home/ajfer6/ab57/anthony-PET/MRI-Deeplearningtest/Matlab/backup/rPET11_low.nii'));
high = mat2gray(niftiread('/home/ajfer6/ab57/anthony-PET/MRI-Deeplearningtest/Matlab/backup/rPET11_high.nii'));
MR =  mat2gray(niftiread('/home/ajfer6/ab57/anthony-PET/MRI-Deeplearningtest/Matlab/backup/rsf11_T1.nii'));
% Rearrange data for axial view and padding from 256*176 to 256*256
high1 = [zeros(256,40,256) high(:,:,:) zeros(256,40,256)];
low1 = [zeros(256,40,256) low(:,:,:) zeros(256,40,256)];
MR1 = [zeros(256,40,256) MR(:,:,:) zeros(256,40,256)];

% for loop for masking images 
for k = 1:256
    high2(:,:,k) = imfill(imbinarize(high1(:,:,k),0.02),'holes').*high1(:,:,k);
    low2(:,:,k) = imfill(imbinarize(low1(:,:,k),0.02),'holes').*low1(:,:,k);
    MR2(:,:,k) = imfill(imbinarize(MR1(:,:,k),0.02),'holes').*low1(:,:,k);
end

%Slices cropped for PET and MR
highc = high2(:,:,55:175);
lowc = low2(:,:,55:175);
MRc = MR2(:,:,55:175);

%%% Low-dose slice - MR slice stack as one array %%
for k = 1:y(3)
    MRI1(:,:,(k*2)-1) = MR(:,:,k);
    MRI1(:,:,k*2) = low(:,:,k);
end

y = size(highc);
%%% Array padded with zeros slices for RGB - G prediction (Multi-Input
%%% strategy) %%
MP = 2*y(3)+4;
H = y(3);
MRIlowf = [zeros(y(1),y(2),2) MRI1 zeros(y(1),y(2),2)];

%%% h5 file creation %%
i = 11;
fname = sprintf('MP%dH.h5',i)
augname = [sprintf('MP%dHR1.h5',i);sprintf('MP%dHR2.h5',i);sprintf('MP%dHR3.h5',i); sprintf('MP%dHT1.h5',i);sprintf('MP%dHT2.h5',i);sprintf('MP%dHT3.h5',i);sprintf('MP%dHT4.h5',i);sprintf('MP%dHT5.h5',i)]

h5create(fname,'/data',[y(1) y(2) MP])
h5create(fname,'/labelImg',[y(1) y(2) H])
h5write(fname, '/data', MRIlowf)
h5write(fname, '/labelImg',highc)

%%% Data Augmentation and h5file creation %%%
RotIN90(:,:,:) = imrotate(MRIlowf(:,:,:), 90,'bilinear','crop');
RotOU90(:,:,:) = imrotate(highc(:,:,:), 90,'bilinear','crop');
RotIN180(:,:,:) = imrotate(MRIlowf(:,:,:), 180,'bilinear','crop');
RotOU180(:,:,:) = imrotate(highc(:,:,:), 180,'bilinear','crop');
RotIN270(:,:,:) = imrotate(MRIlowf(:,:,:), 270,'bilinear','crop');
RotOU270(:,:,:) = imrotate(highc(:,:,:), 270,'bilinear','crop');

T1IN(:,:,:) = imtranslate(MRIlowf,[1.5 1.5]);
T1OU(:,:,:) = imtranslate(highc,[1.5 1.5]);
T2IN(:,:,:) = imtranslate(MRIlowf,[-1.5 -1.5]);
T2OU(:,:,:) = imtranslate(highc,[-1.5 -1.5]);
T3IN(:,:,:) = imtranslate(MRIlowf,[2.5 2.5]);
T3OU(:,:,:) = imtranslate(highc,[2.5 2.5]);
T4IN(:,:,:) = imtranslate(MRIlowf,[-2.5 -2.5]);
T4OU(:,:,:) = imtranslate(highc,[-2.5 -2.5]);
T5IN(:,:,:) = imtranslate(MRIlowf,[4 2.5]);
T5OU(:,:,:) = imtranslate(highc,[4 2.5]);


h5create(augname(1,:),'/data',[y(1) y(2) MP])
h5create(augname(1,:),'/labelImg',[y(1) y(2) H])
h5write(augname(1,:), '/data', RotIN90)
h5write(augname(1,:), '/labelImg',RotOU90)

h5create(augname(2,:),'/data',[y(1) y(2) MP])
h5create(augname(2,:),'/labelImg',[y(1) y(2) H])
h5write(augname(2,:), '/data', RotIN180)
h5write(augname(2,:), '/labelImg',RotOU180)

h5create(augname(3,:),'/data',[y(1) y(2) MP])
h5create(augname(3,:),'/labelImg',[y(1) y(2) H])
h5write(augname(3,:), '/data', RotIN270)
h5write(augname(3,:), '/labelImg',RotOU270)

h5create(augname(4,:),'/data',[y(1) y(2) MP])
h5create(augname(4,:),'/labelImg',[y(1) y(2) H])
h5write(augname(4,:), '/data', T1IN)
h5write(augname(4,:), '/labelImg',T1OU)

h5create(augname(5,:),'/data',[y(1) y(2) MP])
h5create(augname(5,:),'/labelImg',[y(1) y(2) MP])
h5write(augname(5,:), '/data', T2IN)
h5write(augname(5,:), '/labelImg',T2OU)

h5create(augname(6,:),'/data',[y(1) y(2) MP])
h5create(augname(6,:),'/labelImg',[y(1) y(2) H])
h5write(augname(6,:), '/data', T3IN)
h5write(augname(6,:), '/labelImg',T3OU)

h5create(augname(7,:),'/data',[y(1) y(2) MP])
h5create(augname(7,:),'/labelImg',[y(1) y(2) H])
h5write(augname(7,:), '/data', T4IN)
h5write(augname(7,:), '/labelImg',T4OU)
mp
h5create(augname(8,:),'/data',[y(1) y(2) MP])
h5create(augname(8,:),'/labelImg',[y(1) y(2) H])
h5write(augname(8,:), '/data', T5IN)
h5write(augname(8,:), '/labelImg',T5OU)