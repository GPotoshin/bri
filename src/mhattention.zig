const std = @import("std");
const att = @import("attention.zig");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;
const Attention = att.Attention;

pub const MHAttentionHeader = struct {
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
            std.log.err("Can't write version\n", .{});
            return e;
        };
        writer.writeInt(u32, self.type_len, .little) catch |e| {
            std.log.err("Can't write type_size\n", .{});
            return e;
        };
        writer.writeInt(u32, self.heads, .little) catch |e| {
            std.log.err("Can't write heads\n", .{});
            return e;
        };
        writer.writeInt(u32, self.seq_dim, .little) catch |e| {
            std.log.err("Can't write seq_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.ctx_dim, .little) catch |e| {
            std.log.err("Can't write ctx_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.att_dim, .little) catch |e| {
            std.log.err("Can't write att_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.mid_dim, .little) catch |e| {
            std.log.err("Can't write mid_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.out_dim, .little) catch |e| {
            std.log.err("Can't write out_dim\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_seq_len, .little) catch |e| {
            std.log.err("Can't write max_seq_len\n", .{});
            return e;
        };
        writer.writeInt(u32, self.max_ctx_len, .little) catch |e| {
            std.log.err("Can't write seq_dim\n", .{});
            return e;
        };
    }

    pub fn read(self: *Self, reader: anytype) !void {
        self.version = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read version\n", .{});
            return e;
        };
        self.type_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read type\n", .{});
            return e;
        };
        self.heads = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read heads\n", .{});
            return e;
        };
        self.seq_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read seq_dim\n", .{});
            return e;
        };
        self.ctx_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read ctx_dim\n", .{});
            return e;
        };
        self.att_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read att_dim\n", .{});
            return e;
        };
        self.mid_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read mid_dim\n", .{});
            return e;
        };
        self.out_dim = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read out_dim\n", .{});
            return e;
        };
        self.max_seq_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read max_seq_len\n", .{});
            return e;
        };
        self.max_ctx_len = reader.readInt(u32, .little) catch |e| {
            std.log.err("Can't read max_ctx_len\n", .{});
            return e;
        };
    }

    pub fn toAttentionHeader(self: Self) att.AttentionHeader {
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

    test toAttentionHeader {
        const mhheader = MHAttentionHeader {
            .version = 0,
            .type_len = 1,
            .heads = 2,
            .seq_dim = 3,
            .ctx_dim = 4,
            .att_dim = 5,
            .mid_dim = 6,
            .out_dim = 7,
            .max_seq_len = 8,
            .max_ctx_len = 9,
        };
        const header = mhheader.toAttentionHeader();
        try std.testing.expectEqualDeep(att.AttentionHeader {
            .version = 0,
            .type_len = 1,
            .seq_dim = 3,
            .ctx_dim = 4,
            .att_dim = 5,
            .out_dim = 6,
            .max_seq_len = 8,
            .max_ctx_len = 9,
        }, header);
    }
};

pub fn MHAttention(comptime T: type) type {
    return struct {
        header: MHAttentionHeader,

        attentions: []Attention(T),
        att_results: Matrix(T),
        comb_matrix: Matrix(T),
        comb_vect: []T,

        out: Matrix(T),

        const Self = @This();
        pub fn init(allocator: std.mem.Allocator, header: MHAttentionHeader, out: Matrix(T)) !Self {    
            var retval: Self = undefined;

            // maybe change the number of parameters passed?
            if (out.height != header.max_seq_len or out.width != header.out_dim) {
                std.log.err("Wrong dimensions for output matrix for MHAttension\n", .{});
                return error.DataConflict;
            }

            retval.header = header;

            retval.attentions = try allocator.alloc(Attention(T), header.heads);
            retval.att_results = try Matrix(T).init(allocator, header.max_seq_len
                * header.heads, header.mid_dim);
            for (0..header.heads) |i| {
                retval.attentions[i] = try att.Attention(T).init(allocator, .{
                    .version = 0,
                    .type_len = @sizeOf(T),
                    .seq_dim = header.seq_dim,
                    .ctx_dim = header.ctx_dim,
                    .att_dim = header.att_dim,
                    .out_dim = header.mid_dim,
                    .max_seq_len = header.max_seq_len,
                    .max_ctx_len = header.max_ctx_len},
                    retval.att_results.submatrix(@truncate(i*header.max_seq_len),
                    @truncate((i+1)*header.max_seq_len)).?,
                );
            }

            retval.comb_matrix = try Matrix(T).init(allocator, header.out_dim*header.heads,
                header.mid_dim);
            retval.comb_vect = try allocator.alloc(T, header.out_dim);
            retval.out = out;

            return retval;
        }

        pub fn allocateForHeader(self: *Self, allocator: std.mem.Allocator) !void {
            self.attentions = try allocator.alloc(att.Attention(T), self.header.heads);
            for (self.attentions) |*a| {
                a.header = self.header.toAttentionHeader();
                try a.allocateForHeader(allocator);
            }
            self.att_results = try Matrix(T).init(allocator, self.header.heads *
                self.header.max_seq_len, self.header.mid_dim);
            for (self.attentions, 0..) |*a, i| {
                a.out = self.att_results.submatrix(@truncate(i*a.header.max_seq_len),
                    @truncate((i+1)*a.header.max_seq_len)).?;
            }
            self.comb_matrix = try Matrix(T).init(allocator, self.header.out_dim*self.header.heads,
                self.header.mid_dim);
            self.comb_vect = try allocator.alloc(T, self.header.out_dim);
        }

        pub fn destroy(self: *Self, allocator: std.mem.Allocator) void {
            for (self.attentions) |*a| {
                a.destroy(allocator);
            }
            allocator.free(self.attentions);
            self.att_results.destroy(allocator);
            self.comb_matrix.destroy(allocator);
            allocator.free(self.comb_vect);
        }

        pub fn readWeights(self: *Self, reader: anytype) !void {
            for (self.attentions) |*a| {
                a.readWeights(reader) catch |e| {
                    std.log.err("Can't read an attetion block\n", .{});
                    return e;
                };
            }
            self.comb_matrix.read(reader) catch |e| {
                std.log.err("Can't read combination matrix\n", .{});
                return e;
            };
            mtx.readVector(T, reader, self.comb_vect) catch |e| {
                std.log.err("Can't read combination vector\n", .{});
                return e;
            };
        }

        pub fn writeWeights(self: *Self, writer: anytype) !void {
            for (self.attentions) |*a| {
                a.writeWeights(writer) catch |e| {
                    std.log.err("Can't write an attetion block\n", .{});
                    return e;
                };
            }
            self.comb_matrix.write(writer) catch |e| {
                std.log.err("Can't write combination matrix\n", .{});
                return e;
            };
            mtx.writeVector(T, writer, self.comb_vect) catch |e| {
                std.log.err("Can't write combination vector\n", .{});
                return e;
            };
        }

        pub fn fillRandom(self: Self, rand: std.Random, k: T) void {
            for (self.attentions) |a| {
                a.fillRandom(rand, k);
            }
            self.comb_matrix.fillRandom(rand, k);
            mtx.fillVecRandom(T, rand, self.comb_vect, k);
        }

        /// Do not forget to set out matrix
        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            try file.seekTo(0);
            const reader = file.reader();
            var retval: Self = undefined;
            try retval.header.read(reader);
            try retval.allocateForHeader(allocator);
            try retval.readWeights(reader);
            return retval;
        }

        // @AddThreads
        pub fn compute(self: *Self, ctx: Matrix(T), seq: Matrix(T), comptime mask: att.Mask) !void {

            @memset(self.out.toSlice(), 0);
            // Check that the out matrix is the right one
            // remove multiplication
            const transform_size = self.header.out_dim;
            const data_size = self.header.max_seq_len;
            if (seq.height > self.header.max_seq_len or ctx.height > self.header.max_ctx_len) {
                std.log.err("Input is too long!\n", .{});
                return error.IncompatibleObjects;
            }
            for (self.attentions) |*a| {
                try a.compute(ctx, seq, mask);
            }
            
            // this way we can pass seq to the out @TestIt
            for (0..self.attentions.len) |i| {
                const start_transform: u32 = @truncate(transform_size*i);
                const end_transform: u32 = @truncate(transform_size*(i+1));
                const transform = self.comb_matrix.submatrix(start_transform, end_transform).?;

                const start_data: u32 = @truncate(data_size*i);
                const end_data: u32 = @truncate(data_size*(i+1));
                var data = self.att_results.submatrix(start_data, end_data).?;
                data.height = seq.height;

                try mtx.mataddprod(T, transform, data, &self.out);
            }
            try self.out.addRow(self.comb_vect);
        }

        const newTestData = @import("test.zig").Data(T);

        test allocateForHeader {
            const allocator = std.testing.allocator;
            var mhatt: Self = undefined;
            mhatt.header = newTestData.big_mhatt_header;
            try mhatt.allocateForHeader(allocator);
            defer mhatt.destroy(allocator);
        }

        test init {
            const allocator = std.testing.allocator;
            var mhatt = try init(allocator, newTestData.big_mhatt_header,
                newTestData.big_mhatt_out);
            mhatt.destroy(allocator);
        }

        test fillRandom {
            const allocator = std.testing.allocator;
            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();
            var mhatt = try init(allocator, newTestData.big_mhatt_header,
                newTestData.big_mhatt_out);
            defer mhatt.destroy(allocator);

            mhatt.fillRandom(rand, 0.1);
        }

        test writeWeights {
            newTestData.prepareTest();

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .write_only});
            defer file.close();
            const writer = file.writer();

            var mhatt = newTestData.test_mhatt;

            try mhatt.header.write(writer);
            try mhatt.writeWeights(writer);
            try file.setEndPos(try file.getPos());
        }

        test readWeights {
            const allocator = std.testing.allocator;

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .read_only});
            defer file.close();
            const reader = file.reader();
            
            var mhatt: MHAttention(T) = undefined;
            try mhatt.header.read(reader);
            try mhatt.allocateForHeader(allocator);
            defer mhatt.destroy(allocator);
            try mhatt.readWeights(reader);


            // testing header data
            try std.testing.expectEqualDeep(newTestData.test_mhatt_header, mhatt.header);
            try std.testing.expectEqualDeep(newTestData.test_att1_header, mhatt.attentions[2].header);

            // testing wights data
            try std.testing.expectEqualSlices(T, mhatt.attentions[0].query_matrix.toSlice(),
                &newTestData.test_att1_cont1);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].query_vect,
                &newTestData.test_att1_cont2);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].key_matrix.toSlice(),
                &newTestData.test_att1_cont3);
            try std.testing.expectEqualSlices(T, mhatt.attentions[3].key_vect,
                &newTestData.test_att1_cont4);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].value_matrix.toSlice(),
                &newTestData.test_att1_cont5);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].value_vect,
                &newTestData.test_att1_cont6);
            try std.testing.expectEqualSlices(T, mhatt.comb_matrix.toSlice(),
                &newTestData.test_comb_mat_data);
            try std.testing.expectEqualSlices(T, mhatt.comb_vect,
                &newTestData.com_vec_data);
        }

        test initFromFile {
            const allocator = std.testing.allocator;

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .read_only});
            defer file.close();
            var mhatt = try MHAttention(T).initFromFile(allocator, file);
            defer mhatt.destroy(allocator);


            // testing header data
            try std.testing.expectEqualDeep(newTestData.test_mhatt_header, mhatt.header);
            try std.testing.expectEqualDeep(newTestData.test_att1_header, mhatt.attentions[2].header);

            // testing wights data
            try std.testing.expectEqualSlices(T, mhatt.attentions[0].query_matrix.toSlice(),
                &newTestData.test_att1_cont1);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].query_vect,
                &newTestData.test_att1_cont2);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].key_matrix.toSlice(),
                &newTestData.test_att1_cont3);
            try std.testing.expectEqualSlices(T, mhatt.attentions[3].key_vect,
                &newTestData.test_att1_cont4);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].value_matrix.toSlice(),
                &newTestData.test_att1_cont5);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].value_vect,
                &newTestData.test_att1_cont6);
            try std.testing.expectEqualSlices(T, mhatt.comb_matrix.toSlice(),
                &newTestData.test_comb_mat_data);
            try std.testing.expectEqualSlices(T, mhatt.comb_vect,
                &newTestData.com_vec_data);
        }

        test compute {
            const allocator = std.testing.allocator;

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .read_only});
            defer file.close();
            var mhatt = try MHAttention(T).initFromFile(allocator, file);
            defer mhatt.destroy(allocator);
            mhatt.out = newTestData.test_out_matrix;

            try mhatt.compute(newTestData.test_ctx, newTestData.test_seq, .bidirectional);

            // right results
            try newTestData.compare_delta(&newTestData.test_att1_answer_data, mhatt.out.toSlice(), 0.00001);
        }
    };
}

comptime {
    _ = MHAttention(f32);
}
