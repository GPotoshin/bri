/// seq2seq transformer
const std = @import("std");
const parsing = @import("parsing.zig");
const enc = @import("encoder.zig");
const dec = @import("decoder.zig");
const mtx = @import("matrix.zig");

const Encoder = enc.Encoder;
const Decoder = dec.Decoder;
const Matrix = mtx.Matrix;

pub const EDHeader = struct {
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
    tokens_count: u32,

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

    pub fn toDHeader(self: Self) dec.DecoderHeader {
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

    pub fn toEHeader(self: Self) enc.EncoderHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .heads = self.heads,
            .ctx_dim = self.ctx_dim,
            .att_dim = self.att_dim,
            .mid_dim = self.mid_dim,
            .mlp_dim = self.mlp_dim,
            .max_ctx_len = self.max_ctx_len,
        };
    }
};

pub fn EDTransformer(comptime T: type) type {
    return struct {
        header: EDHeader,

        encoder: Encoder(T),
        decoder: Decoder(T),

        unembedding_matrix: Matrix(T),
        out: Matrix(T),

        const Self = @This();

        // Attention, embeding is reused as unembedding for now
        pub fn initFromReader(allocator: std.mem.Allocator, reader: anytype, embedding: Matrix(T)) !Self {
            var retval: Self = undefined;
            try retval.header.read(reader);
            retval.header.tokens_count = embedding.height;
            if (embedding.width != retval.header.seq_dim) {
                std.log.err("embedding dimension should be the same as sequence dimention\n", .{});
                return error.DataConflict;
            }
            // I've heard that embedding and unembedding can have the same configuration and it's quite logical
            retval.unembedding_matrix = embedding;
            retval.allocateForHeader(allocator);
            retval.readWeights(reader);
            return retval;
        }

        pub fn writeToFile(self: Self, file: std.fs.File) !void {
            try file.seekTo(0);
            const writer = file.writer();
            try self.header.write(writer);
            try self.writeWeights(writer);
            try file.setEndPos(try file.getPos());
        }

        pub fn allocateForHeader(self: Self, allocator: std.mem.Allocator) !void {
            self.encoder.header = self.header.toEHeader();
            self.decoder.header = self.header.toDHeader();
            try self.encoder.allocateForHeader(allocator);
            try self.decoder.allocateForHeader(allocator);
            self.out = try Matrix(T).init(allocator, self.header.max_seq_len, self.header.tokens_count);
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            self.encoder.fillRandom(rand, k);
            self.decoder.fillRandom(rand, k);
        }

        /// Attention! The content of ctx and seq is changed! Was it a good design choice?
        pub fn compute(self: Self, ctx: Matrix(T), seq: Matrix(T)) !void {
            self.encoder.apply(ctx);
            self.decoder.apply(ctx, seq);
            mtx.matprod(T, self.unembedding_matrix, self.seq, self.out);
            mtx.softmax(T, self.seq);
        }

        pub fn writeWeights(self: Self, writer: anytype) !void {
            self.encoder.writeWeights(writer);
            self.decoder.writeWeights(writer);
        }

        pub fn readWeights(self: Self, reader: anytype) !void {
            self.encoder.readWeights(reader);
            self.decoder.readWeights(reader);
        }
    };
}
