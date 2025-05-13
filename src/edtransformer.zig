/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");
const enc = @import("encoder.zig");
const dec = @import("decoder.zig");
const mtx = @import("matrix.zig");

const Encoder = enc.Encoder;
const Decoder = dec.Decoder;
const Matrix = mtx.Matrix;


pub fn EDTransformer(comptime T: type) type {
    return struct {
        encoder: Encoder(T),
        decoder: Decoder(T),

        const Self = @This();

    };
}
