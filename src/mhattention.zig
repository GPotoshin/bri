const std = @import("std");
const att = @import("attention.zig");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;

const MHAttentionHeader = struct {
    version: u32,
    type_len: u32,
    heads: u32,
    seq_dim: u32,
    ctx_dim: u32,
    att_dim: u32,
    mid_dim: u32,
    out_dim: u32,
    max_seq_len: u32,
    max_ctx_len: u32,

    const Self = @This();
    pub fn write(self: Self, writer: anytype) !void {
        writer.writeInt(u32, self.version, .little) catch |e| {
            std.debug.print("Can't write version\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.debug.print("Can't write type_size\n", .{});
            return e;
        };
        writer.writeInt(u32, self.heads, .little) catch |e| {
            std.debug.print("Can't write heads\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.debug.print("Can't write seq_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.debug.print("Can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.debug.print("Can't write att_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.debug.print("Can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.out_dim, .little) catch |e| {
            std.debug.print("Can't write out_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.debug.print("Can't write max_seq_len\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.debug.print("Can't write seq_dim\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read type\n", .{});
            return e;
        };
        self.heads = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read heads\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read seq_dim\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read ctx_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read att_dim\n", .{});
            return e;
        };
        self.mid_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read mid_dim\n", .{});
            return e;
        };
        self.out_dim = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read out_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read max_seq_len\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.debug.print("Can't read max_ctx_len\n", .{});
            return e;
        };
    }

    pub fn toAttentionHeader(self: Self) AttentionHeader {
        return .{
            .version = self.version,
            .type_len = self.type_len,
            .seq_dim = self.seq_dim,
            .ctx_dim = self.ctx_dim,
            .att_dim = self.att_dim,
            .out_dim = self.mid_dim,
            .max_seq_len = self.max_seq_len,
            .max_ctx_len = self.max_ctx_len,
        };
    }
};

pub fn MHAttention(comptime T: type) type {
    return struct {
        header: MHAttentionHeader,

        attentions: []att.Attention(T),
        att_results: mtx.Matrix(T),
        comb_matrix: mtx.Matrix(T),
        comb_vect: []T,


        out: Matrix(T),

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, header: MHAttentionHeader, out: Matrix(T)) Self {    
            var retval: Self = undefined;

            // maybe change the number of parameters passed?
            if (out.height != header.max_seq_len or out.width != header.out_dim) {
                std.debug.print("Wrong dimensions for output matrix for MHAttension\n", .{});
                return error.DataConflict;
            }

            retval.header = header;

            retval.attentions = try allocator.alloc(T, hyper_param.heads);
            retval.att_results = try Matrix(T).init(allocator, hyper_param.max_seq_len
                * hyper_param.heads, hyper_param.mid_dim);
            for (0..hyper_param.heads) |i| {
                retval.attentions[i] = try Attention(T).init(allocator, .{
                    .version = 0,
                    .type_len = @sizeOf(T),
                    .seq_dim = hyper_param.seq_dim,
                    .ctx_dim = hyper_param.ctx_dim,
                    .att_dim = hyper_param.att_dim,
                    .out_dim = hyper_param.mid_dim,
                    .max_seq_len = hyper_param.max_seq_len,
                    .max_ctx_len = hyper_param.max_ctx_len},
                    retval.att_results.submatrix(i*hyper_param.max_seq_len,
                        (i+1)*hyper_param.max_seq_len),
                );
            }

            retval.comb_matrix = try Matrix(T).init(allocator, hyper_param.out_dim,
                hyper_param.mid_dim*hyper_param.heads);
            retval.comb_vect = try allocator.alloc(T, hyper_param.out_dim);
            retval.out = out;

            return retval;
        }

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.attentions = try allocator.alloc(att.Attention(T), header);
            for (self.attentions) |*a| {
                a.header = self.header.toAttentionHeader();
                try a.allocateForHeader(allocator);
            }
            self.att_results = try Matrix(T).init(allocator, self.header.heads *
                self.header.max_seq_len, self.header.mid_dim);
            self.comb_matrix = try Matrix(T).init(allocator, self.header.out_dim,
                self.header.heads*self.header.mid_dim);
            self.comb_vect = try allocator.alloc(T, self.header.out_dim);
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            for (self.attentions) |*a| {
                a.readWeights(reader) catch |e| {
                    std.debug.print("Can't read an attetion block\n", .{});
                    return e;
                };
            }
            self.comb_matrix.read(reader) catch |e| {
                std.debug.print("Can't read combination matrix\n" ,{});
                return e;
            };
            mtx.readVector(T, reader, self.comb_vect) catch |e| {
                std.debug.print("Can't read combination vector\n", .{});
                return e;
            };
        }

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            try file.seekTo(0);
            var reader = file.reader();
            var retval: Self = undefined;
            retval.header.read(reader);
            retval.allocateForHeader(allocator);
            retval.readWeights(reader);
        }
    };
}
