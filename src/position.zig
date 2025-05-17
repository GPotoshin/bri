const std = @import("std");

pub fn addPositionInformation(arr: []f32, pos: usize, n: f32) void {
    const dim_f: f32 = @floatFromInt(arr.len);
    const pos_f: f32 = @floatFromInt(pos);
    const factor = std.math.pow(f32, n, 2.0/dim_f);

    for (0..@divFloor(arr.len, 2)) |i| {
        const i_f: f32 = @floatFromInt(i);
        arr[2*i] += @sin(pos_f/std.math.pow(f32, factor, i_f));
        arr[2*i+1] += @cos(pos_f/std.math.pow(f32, factor, i_f));
    }

}
