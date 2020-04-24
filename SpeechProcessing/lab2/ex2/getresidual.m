function [residual, A] = getresidual(x, p, window_len, window_spacing)
    A = v_lpcauto(x, p, [window_spacing, window_len, 0]);
    residual = zeros(size(x));
    zf = zeros(1, p);
    for i = 1:size(A, 1)
        n = (i==1)*p+1 + (i>1)*((i-0.5)*window_spacing):min((i+0.5)*window_spacing,size(x, 1));
        [residual(n), zf] = filter(A(i,:), 1, x(n), zf);
    end
end
