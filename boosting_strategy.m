function [rhoM,rhoP, mask_p] = boosting_strategy(cover_Path, QF, distortion)
% distortion: true is UERD, false is JUNIWARD
wetConst = 10^13;

img_ori = jpeg_read(cover_Path);
cover_COEFF = img_ori.coef_arrays{1};
q = img_ori.quant_tables{1};

%% Reduce the Cover Elements
img_size = size(img_ori.coef_arrays{1});
qmatrix = repmat(q, img_size/8);
mask_quant_step = qmatrix<22;
mask_p = mask_quant_step;


[rc_COEFF, ~] = recompress(img_ori, QF, [cover_Path, 'a']);
truncation = img_ori.coef_arrays{1} - rc_COEFF;


if distortion
    [rhoP1,rhoM1] = J_UNIWARD_Asy_cost(cover_Path);
else
    [rhoP1,rhoM1] = J_UERD_Asy_cost(img_ori, cover_Path);
end



fun = @(x) idct2(x.data.*q);
spa_uq = blockproc(cover_COEFF,[8 8],fun);

rhoP = rhoP1;
rhoM = rhoM1;

%% Recompression Attacks Errors
%% Lower bound of cost optimization
overflow_pixel = zeros(size(spa_uq));
overflow_pixel(spa_uq>127) = 1;
overflow_pixel(spa_uq<-128) = 1;
fun = @(x) sum(sum(abs(x.data)))*ones(8,8);
overflow_blk = blockproc(overflow_pixel,[8 8],fun);

rhoP(overflow_blk~=0) = wetConst;
rhoM(overflow_blk~=0) = wetConst;


threshold =0; %
fun = @(block_struct)truncated_boost(block_struct.data, threshold);
truncated = blockproc(truncation, [8, 8], fun);
rhoP(truncated ~= 0) = truncated(truncated ~= 0) .* rhoP(truncated ~= 0);
rhoM(truncated ~= 0) = truncated(truncated ~= 0) .* rhoM(truncated ~= 0);



mask_tmp = mask_p;
fun = @(block_struct)processBlock(block_struct.data);

% 去掉DCT系数错误块的模拟嵌入
c = cat(3, mask_tmp, truncation);
mask_p1 = blockproc(c, [8, 8], fun);

mask_p1_p = mask_p1;
mask_p1_n = -mask_p1;




simul_stego_p_coeff = rc_COEFF + mask_p1_p;
simul_stego_n_coeff = rc_COEFF + mask_p1_n;

fun = @(x) idct2(x.data.*q);
simul_spa_p = blockproc(simul_stego_p_coeff,[8 8],fun);
simul_spa_n = blockproc(simul_stego_n_coeff,[8 8],fun);


overflow_piexl2 = zeros(size(simul_spa_p));

overflow_piexl2(simul_spa_p>127) = 1;
overflow_piexl2(simul_spa_n<-128) = 1;

threshold = 1; %
fun = @(block_struct)truncated_boost(block_struct.data, threshold);
truncated = blockproc(overflow_piexl2, [8, 8], fun);
rhoP(truncated ~= 0) = truncated(truncated ~= 0) .* rhoP(truncated ~= 0);
rhoM(truncated ~= 0) = truncated(truncated ~= 0) .* rhoM(truncated ~= 0);

