
% Formants - taken from wavesurfer
global f1_means f2_means f3_means f4_means;
global f1_stds f2_stds f3_stds f4_stds;
%           i     @     E     e     6     a     u     o     O
f1_means = [225,  352,  489,  362,  559,  779,  378,  428,  568];
f2_means = [2280, 1768, 1910, 1956, 1586, 1179, 931,  969,  1036];
f3_means = [3129, 2569, 2724, 2815, 2676, 2682, 2202, 2385, 2414];
f4_means = [3770, 3427, 4016, 4155, 4068, 4141, 2869, 3110, 3061];
f1_stds  = [22,   32,   11,   27,   22,   57,   32,   10,   6];
f2_stds  = [30,   12,   10,   27,   73,   25,   17,   30,   10];
f3_stds  = [34,   41,   22,   41,   20,   24,   54,   43,   26];
f4_stds  = [176,  98,   57,   61,   70,   60,   65,   140,  30];

[x_fixed, fs] = generate_vowel(7, 110, 110, 0.5, 0.005, 0.005);
audiowrite("ex4/formant_synthesis_fixed.wav", x_fixed, fs);
% sound(x_fixed, fs);

[x_var, fs] = generate_vowel(7, 90, 150, 0.5, 0.001, 0.01);
audiowrite("ex4/formant_synthesis_var.wav", x_var, fs);
% sound(x_var, fs);

[x_two, fs] = generate_two(6, 9, 100, 150, 0.5, 0.01, 0.02);
audiowrite("ex4/formant_synthesis_two.wav", x_two, fs);
sound(x_two, fs);

function [x_recon, fs] = generate_two(vowel1, vowel2, f0_start, f0_end, duration, intensity_start, intensity_end)
    [x1, fs] = generate_vowel(vowel1, f0_start, f0_end, duration, intensity_start, intensity_end);
    [x2, fs] = generate_vowel(vowel2, f0_start, f0_end, duration, intensity_start, intensity_end);

    x_recon = cat(1, x1, x2);
end

function [x_recon, fs] = generate_vowel(vowel, f0_start, f0_end, duration, intensity_start, intensity_end)

    global f1_means f2_means f3_means f4_means;
    global f1_stds f2_stds f3_stds f4_stds;
    fs = 16000;  % sampling frequency
    n = 0.02;
    window_len = n * fs;
    window_spacing = window_len / 2;
    
    n = round(fs * duration);                          % number of samples
    sz = size(0: window_spacing : n - window_len, 2);  % number of rows in A
    A = zeros(sz, 9);
    for j = 1: sz
        f1 = normrnd(f1_means(vowel),  f1_stds(vowel));
        f2 = normrnd(f2_means(vowel),  f2_stds(vowel));
        f3 = normrnd(f3_means(vowel),  f3_stds(vowel));
        f4 = normrnd(f4_means(vowel),  f4_stds(vowel));

        roots = [z(f1, fs), z(f2, fs), z(f3, fs), z(f4, fs)];
        A(j, :) = poly([roots, conj(roots)]);
    end

    nb_iterations = round(n / mean(f0_start, f0_end));
    delta = (f0_end - f0_start) / nb_iterations;
    delta_intensity = (intensity_end - intensity_start) / nb_iterations;
    
    p = 8;
    artificial_excitation = zeros(n, 1);
    start = 1;
    f0 = f0_start;
    intensity = intensity_start;
    for i = 1 : nb_iterations
        f0 = f0 + delta;
        intensity = intensity + delta_intensity;

        artificial_excitation(start) = intensity;
        interval = round( fs / f0 );
        start = start + interval;
    end

    x_recon = getx(artificial_excitation, A, p, window_spacing);
end


function zk = z(fk, Fs)
    % bk = pole magnitude of 0.95, in order to have poles near the unit circumference
    zk = 0.95 * exp( 1i * 2 * pi * fk / Fs );
end
