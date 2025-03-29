/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");
const mhatt = @import("mhattention.zig");
const MHAttention = mhatt.MHAttention;

// is that a good idea?


pub fn EDTransformer(comptime T: type) type {
    return struct {
        mhatt1: MHAttention(T),

        const Self = @This();

        pub fn calculate(self: Self) void {
            


        }
    };
}
