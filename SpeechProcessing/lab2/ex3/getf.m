function f = getf(w, Fs, threshold)
    n_samples = size(w, 1);
    R = xcorr(w);
    
    if(R(n_samples) < 0.01)
        f = 0;
        return;
    end
    
    Rn = R/R(n_samples);

    [peak, locs] = findpeaks(Rn);

    central_peak = n_samples;
    
    fmin = 60; 
    fmax = 160;

    mask = (peak > threshold & locs > central_peak) & (locs >= central_peak + Fs/fmax) & (locs <= central_peak + Fs/fmin);

    p = peak(mask);
    l = locs(mask);
    
    [~, best_pic] = max(p);
    
    if isempty(l)
        f = 0;
    else
        f = Fs/(l(best_pic) - central_peak);
    end
end