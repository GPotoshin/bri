const std = @import("std");
const mhatt = @import("mhattention.zig");
const ln = @import("layernorm.zig");
const mlp = @import("multilayerpreceptron.zig");
const mtx = @import("matrix.zig");

const Matrix = mtx.Matrix;
const MHAttention = mhatt.MHAttention;
const LayerNorm = ln.LayerNorm;
const MultilayerPreceptron = mlp.MultilayerPreceptron;

pub const DecodeLayerHeader = struct {
    version: u32,
    type_len: u32,
    heads: u32,
    ctx_dim: u32,
    seq_dim: u32,
    att_dim: u32,
    mid_dim: u32,
    mlp_dim: u32,
    max_ctx_len: u32,
    max_seq_len: u32,

    const Self = @This();
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.log.err("can't write version\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("can't write type_size\n", .{});
            return e;
        };
        writer.writeInt(u32, self.heads, .little) catch |e| {
            std.log.err("can't write heads\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.log.err("can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.log.err("can't write seq_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.log.err("can't write att_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mlp_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.log.err("can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("can't write seq_dim\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read type_size\n", .{});
            return e;
        };
        self.heads = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read heads\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read ctx_dim\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read seq_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read att_dim\n", .{});
            return e;
        };
        self.mid_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.mlp_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read ctx_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read seq_dim\n", .{});
            return e;
        };
    }


    pub fn to1stMHAttentionHeader(self: Self) mhatt.MHAttentionHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .heads = self.heads,
            .ctx_dim = self.seq_dim,
            .seq_dim = self.seq_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .out_dim = self.seq_dim,
            .max_ctx_len = self.max_seq_len,
            .max_seq_len = self.max_seq_len,
        };
    }

    pub fn to2ndMHAttentionHeader(self: Self) mhatt.MHAttentionHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .heads = self.heads,
            .ctx_dim = self.ctx_dim,
            .seq_dim = self.seq_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .out_dim = self.seq_dim,
            .max_ctx_len = self.max_ctx_len,
            .max_seq_len = self.max_seq_len,
        };
    }

    pub fn toMLPHeader(self: Self) mlp.MultilayerPreceptronHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .in_dim = self.seq_dim,
            .mid_dim = self.mlp_dim,
            .out_dim = self.seq_dim,
            .max_seq_len = self.max_seq_len,
        };
    }

};

pub fn DecodeLayer(comptime T: type) type {
    return struct {
        header: DecodeLayerHeader,

        mhattention1: MHAttention(T),
        layer_norm1: LayerNorm(T),
        mhattention2: MHAttention(T),
        layer_norm2: LayerNorm(T),
        preceptron: MultilayerPreceptron(T),
        layer_norm3: LayerNorm(T),

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, header: DecodeLayerHeader) !Self {
            var retval: Self = undefined;
            retval.header = header;
            try retval.allocateForHeader(allocator);
            return retval;
        }

        pub fn copyValuesFrom(dest: *Self, source: Self) !void {
            try dest.mhattention1.copyValuesFrom(source.mhattention1);
            try dest.layer_norm1.copyValuesFrom(source.layer_norm1);
            try dest.mhattention2.copyValuesFrom(source.mhattention2);
            try dest.layer_norm2.copyValuesFrom(source.layer_norm2);
            try dest.preceptron.copyValuesFrom(source.preceptron);
            try dest.layer_norm3.copyValuesFrom(source.layer_norm3);
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            self.mhattention1.fillRandom(rand, k);
            self.layer_norm1.fillRandom(rand, k);
            self.mhattention2.fillRandom(rand, k);
            self.layer_norm2.fillRandom(rand, k);
            self.preceptron.fillRandom(rand, k);
            self.layer_norm3.fillRandom(rand, k);
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (
                std.meta.eql(self.header, expected.header) and
                self.mhattention1.isEqualTo(expected.mhattention1, delta) and
                self.layer_norm1.isEqualTo(expected.layer_norm1, delta) and
                self.mhattention2.isEqualTo(expected.mhattention2, delta) and
                self.layer_norm2.isEqualTo(expected.layer_norm2, delta) and
                self.preceptron.isEqualTo(expected.preceptron, delta) and
                self.layer_norm3.isEqualTo(expected.layer_norm3, delta)
            ) {
                return true;
            }
            return false;
        }

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.mhattention1 = try MHAttention(T).init(allocator,
                self.header.to1stMHAttentionHeader(), Matrix(T).empty);
            try self.mhattention1.allocateOut(allocator);
            self.layer_norm1 = try LayerNorm(T).init(allocator, self.header.seq_dim);
            self.mhattention2 = try MHAttention(T).init(allocator,
                self.header.to2ndMHAttentionHeader(), Matrix(T).empty);
            try self.mhattention2.allocateOut(allocator);
            self.layer_norm2 = try LayerNorm(T).init(allocator, self.header.seq_dim);
            self.preceptron = try MultilayerPreceptron(T).init(allocator,
                self.header.toMLPHeader());
            try self.preceptron.allocateOut(allocator);
            self.layer_norm3 = try LayerNorm(T).init(allocator, self.header.seq_dim);
        }

        /// if you do not use arenas for some reason
        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            self.mhattention1.destroy(allocator);
            self.layer_norm1.destroy(allocator);
            self.mhattention2.destroy(allocator);
            self.layer_norm2.destroy(allocator);
            self.preceptron.destroy(allocator);
            self.layer_norm3.destroy(allocator);
        }

        // @Test: do we really need here self: *Self?
        pub fn apply(self: *Self, ctx: Matrix(T), seq: Matrix(T)) !void {
            try self.mhattention1.compute(seq, seq, .unidirectional);   
            try seq.add(self.mhattention1.out);
            self.layer_norm1.apply(seq);
            try self.mhattention2.compute(ctx, seq, .bidirectional);   
            try seq.add(self.mhattention2.out);
            self.layer_norm2.apply(seq);
            try self.preceptron.compute(seq);
            try seq.add(self.preceptron.out);
            self.layer_norm3.apply(seq);
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            try self.mhattention1.writeWeights(writer);
            try self.layer_norm1.writeWeights(writer);
            try self.mhattention2.writeWeights(writer);
            try self.layer_norm2.writeWeights(writer);
            try self.preceptron.writeWeights(writer);
            try self.layer_norm3.writeWeights(writer);
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            try self.mhattention1.readWeights(reader);
            try self.layer_norm1.readWeights(reader);
            try self.mhattention2.readWeights(reader);
            try self.layer_norm2.readWeights(reader);
            try self.preceptron.readWeights(reader);
            try self.layer_norm3.readWeights(reader);
        }

        test readWeights {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();
            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();

            const header = DecodeLayerHeader {
                .version = 1,
                .type_len = @sizeOf(T),
                .heads = 2,
                .seq_dim = 3,
                .ctx_dim = 4,
                .att_dim = 5,
                .mid_dim = 6,
                .mlp_dim = 7,
                .max_seq_len = 8,
                .max_ctx_len = 9,
            };

            var decodeLayer = try Self.init(allocator, header);
            decodeLayer.fillRandom(rand, 0.01);

            var file = try std.fs.cwd().openFile("test_files/test_decodelayer", .{.mode = .write_only});
            const writer = file.writer();
            try decodeLayer.header.write(writer);
            try decodeLayer.writeWeights(writer);
            try file.setEndPos(try file.getPos());
            file.close();

            var new_decodeLayer: DecodeLayer(T) = undefined;
            file = try std.fs.cwd().openFile("test_files/test_decodelayer", .{.mode = .read_only});
            const reader = file.reader();
            try new_decodeLayer.header.read(reader);
            try std.testing.expectEqualDeep(header, new_decodeLayer.header);

            try new_decodeLayer.allocateForHeader(allocator);

            try new_decodeLayer.readWeights(reader);
            try std.testing.expect(new_decodeLayer.isEqualTo(decodeLayer, 0.0001));
        }

        // I suppose it's fine.
        test apply {
            var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer arena.deinit();
            const allocator = arena.allocator();
            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();

            const header = DecodeLayerHeader {
                .version = 1,
                .type_len = @sizeOf(T),
                .heads = 2,
                .seq_dim = 3,
                .ctx_dim = 4,
                .att_dim = 5,
                .mid_dim = 6,
                .mlp_dim = 7,
                .max_seq_len = 8,
                .max_ctx_len = 9,
            };

            var ctx_data: [4*2]T = undefined;
            var seq_data: [3*3]T = undefined;

            mtx.fillVecRandom(T, rand, &ctx_data, 1);
            mtx.fillVecRandom(T, rand, &seq_data, 1);

            const ctx = Matrix(T) {
                .height = 2,
                .width = 4,
                .ptr = &ctx_data,
            };
            const seq = Matrix(T) {
                .height = 3,
                .width = 3,
                .ptr = &seq_data,
            };

            var decodeLayer = try Self.init(allocator, header);
            decodeLayer.fillRandom(rand, 0.01);
            try decodeLayer.apply(ctx, seq);
        }
    };
}

comptime {
    _ = DecodeLayer(f32);
}

pub const DecoderHeader = struct {
    version: u32,
    type_len: u32,
    layers: u32,
    heads: u32,
    ctx_dim: u32,
    seq_dim: u32,
    att_dim: u32,
    mid_dim: u32,
    mlp_dim: u32,
    max_ctx_len: u32,
    max_seq_len: u32,

    const Self = @This();
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.log.err("can't write version\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("can't write type_size\n", .{});
            return e;
        };
        writer.writeInt(u32, self.layers, .little) catch |e| {
            std.log.err("can't write layers\n", .{});
            return e;
        };
        writer.writeInt(u32, self.heads, .little) catch |e| {
            std.log.err("can't write heads\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.log.err("can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.log.err("can't write seq_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.log.err("can't write att_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mlp_dim, .little) catch |e| {
            std.log.err("can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.log.err("can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("can't write seq_dim\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read type_size\n", .{});
            return e;
        };
        self.layers = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read layers\n", .{});
            return e;
        };
        self.heads = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read heads\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read ctx_dim\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read seq_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read att_dim\n", .{});
            return e;
        };
        self.mid_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.mlp_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read mid_dim\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read ctx_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("can't read seq_dim\n", .{});
            return e;
        };
    }

    pub fn toDLHeader(self: Self) DecodeLayerHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .heads = self.heads,
            .ctx_dim = self.ctx_dim,
            .seq_dim = self.seq_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .mlp_dim = self.mlp_dim,
            .max_ctx_len = self.max_ctx_len,
            .max_seq_len = self.max_seq_len,
        };
    }
};
pub fn Decoder(comptime T: type) type {
    return struct {
        header: DecoderHeader,

        layers: []DecodeLayer(T),

        const Self = @This();

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.layers = try allocator.alloc(DecodeLayer(T), self.header.layers);
            for (self.layers) |*layer| {
                layer.header = self.header.toDLHeader();
                layer.allocateForHeader();
            }
        }

        pub fn apply(self: Self, ctx: Matrix(T), seq: Matrix(T)) !void {
            for (self.layers) |layer| {
                layer.apply(ctx, seq);
            }
        }

        pub fn init(allocator: std.mem.Allocator, header: DecoderHeader) !Self {
            var retval: Self = undefined;
            retval.header = header;
            try retval.allocateForHeader(allocator);
            return retval;
        }

        pub fn copyValuesFrom(dest: *Self, source: Self) !void {
            if (std.meta.eql(dest.header, source.header)) { // attention! not all fields should be equal
                return error.IncompatibleObjects;
            }
            for (dest.layers, source.layers) |dlayer, slayer| {
                dlayer.copyValuesFrom(slayer);
            }
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            for (self.layers) |layer| {
                layer.fillRandom(rand, k);
            }
        }

        pub fn isEqualTo(self: Self, expected: Self, delta: T) bool {
            if (std.meta.eql(self.header, expected.header) and self.layers.len == expected.layers.len) {
                for (self.layers, expected.layers) |slayer, elayer| {
                    if (!slayer.isEqualTo(elayer, delta)) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            for (self.layers) |layer| {
                layer.destroy(allocator);
            }
            allocator.free(self.layers);
            self.layers = null;
        }
        
        pub fn writeWeights(self: Self, writer: anytype) !void {
            if (self.layers.len != self.header.layers) {
                return error.IncompatibleObjects;
            }

            for (self.layers) |layer| {
                try layer.read.writeWeights(writer);
            }
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            if (self.layers.len != self.header.layers) {
                return error.IncompatibleObjects;
            }

            for (self.layers) |layer| {
                try layer.read.readWeights(reader);
            }
        }
    };
}
