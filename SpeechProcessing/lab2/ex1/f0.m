[x, Fs] = audioread("vowels_87118.wav");

n = 0.02;
n_samples = n*Fs;
n_samples1 = n/2*Fs;

f = [];

for i = n_samples:n_samples1:size(x, 1)-n_samples
    w = x(i-n_samples/2:i+n_samples/2-1);
    f(size(f, 1)+1, 1) = getf(w, Fs, 0.30);
end

fid = fopen('vowels_87118.myfo', 'w');
for row = 1:length(f)
    fprintf(fid, '%f\n', f(row));
end
fclose(fid);

sprintf("Average F0: %f Hz", sum(f)/sum(f~=0))