[x, Fs] = audioread("birthdate_87118.wav");

n = 0.02;
window_len = n * Fs;
window_spacing = window_len / 2;
p = 16;

[residual, A] = getresidual(x, p, window_len, window_spacing);
artificial_excitation = normrnd(mean(residual(abs(residual) < 0.0005)), std(residual(abs(residual) < 0.0005)), size(residual));  % white noise for unvoiced parts
% artificial_excitation = zeros(size(residual));

last_nonzero = -10000;
for i = window_spacing + 1 : window_spacing : size(x, 1) - window_spacing
    l = i - window_spacing;
    r = i + window_spacing - 1;
    w = x(l : r);
    f0 = getf(w, Fs, 0.30);
    % f0 = 250;
    l = i - window_spacing / 2;
    r = i + window_spacing / 2 - 1;
 
    % Add periodic_impulse_train to excitation if it's a voiced part
    if( f0 ~= 0 )
        artificial_excitation( l: r ) = 0;
        period = round( Fs / f0 );
        start = max( last_nonzero + period, l );
 
        artificial_excitation( start : period : r ) = mean(residual(residual > 0.005));
        last_nonzero = find(artificial_excitation(1 : r), 1, 'last');
        % fprintf("period=%d, range=%d: [%d, %d] => %d added (%d last)\n", period, r - start, start, r, size(find(artificial_excitation(l : r)), 1), last_nonzero);
    end
end

artificial_excitation = artificial_excitation * sqrt(residual' * residual) / sqrt( artificial_excitation' * artificial_excitation);
x_recon = getx(artificial_excitation, A, p, window_spacing);

audiowrite("ex3/birthdate_87118_voc.wav", x_recon, Fs);
