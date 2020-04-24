function x = getx(residual, A, p, window_spacing)
    x = zeros(size(residual));
    zf = zeros(1, p);
    
    for i = 1:size(A, 1)
        n = (i==1)*p+1 + (i>1)*((i-0.5)*window_spacing):min((i+0.5)*window_spacing,size(x, 1));
        [x(n), zf] = filter(1, A(i, :), residual(n), zf);
    end
end