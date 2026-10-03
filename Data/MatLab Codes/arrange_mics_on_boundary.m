function MicCoord = arrange_mics_on_boundary(L, z_fixed, N, margin)
% L        : 1x3 vector specifying room dimensions [Lx, Ly, Lz]
% z_fixed  : fixed z-level where microphones are placed
% N        : total number of microphones to place
% margin   : distance to keep from the wall boundaries (e.g., 0.05)

% Effective room dimensions after applying margin
Lx = L(1) - 2 * margin;
Ly = L(2) - 2 * margin;

% Total perimeter of the inner rectangle (on XY plane)
perimeter = 2 * (Lx + Ly);

% Uniform spacing along the perimeter
d = perimeter / N;

% Generate cumulative distance positions
s = (0:N-1) * d;

MicCoord = zeros(N, 3);  % preallocate

for i = 1:N
    si = mod(s(i)+eps, perimeter);  % +eps for numerical safety

    if si < Lx
        x = si;
        y = 0;
    elseif si < Lx + Ly
        x = Lx;
        y = si - Lx;
    elseif si < 2 * Lx + Ly
        x = Lx - (si - (Lx + Ly));
        y = Ly;
    else
        x = 0;
        y = Ly - (si - (2 * Lx + Ly));
    end

    % Shift entire layout by +margin to stay inside the room
    MicCoord(i, :) = [x + margin, y + margin, z_fixed];
end
end
