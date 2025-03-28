const std = @import("std");
const att = @import("attention.zig");
const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;
const Attention = att.Attention;

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

            retval.comb_matrix = try Matrix(T).init(allocator, header.out_dim,
                header.mid_dim*header.heads);
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
            self.comb_matrix = try Matrix(T).init(allocator, self.header.out_dim,
                self.header.heads*self.header.mid_dim);
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

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            try file.seekTo(0);
            const reader = file.reader();
            var retval: Self = undefined;
            retval.header.read(reader);
            retval.allocateForHeader(allocator);
            retval.readWeights(reader);
        }

        pub fn writeToFile(self: Self, file: std.fs.File) !void {
            try file.seekTo(0);
            const writer = file.writer();
            self.header.write(writer);
            self.writeWeights(writer);
            try file.setEndPos(try file.getPos());
        }

        const testData = struct {
            const header = MHAttentionHeader {
                .version = 0,
                .type_len = @sizeOf(T),
                .heads = 4,
                .seq_dim = 1024,
                .ctx_dim = 1024,
                .att_dim = 1024,
                .mid_dim = 1024,
                .out_dim = 1024,
                .max_seq_len = 1024,
                .max_ctx_len = 1024,
            };
            pub const header2 = MHAttentionHeader {
                .version = 0,
                .type_len = @sizeOf(T),
                .heads = 4,
                .seq_dim = 1,
                .ctx_dim = 2,
                .att_dim = 3,
                .mid_dim = 4,
                .out_dim = 5,
                .max_seq_len = 7,
                .max_ctx_len = 8,
            };

            var comb_mat_data = [_]T {
                0.1, 0.2, 0.3, 0.4,
                0.1, 0.2, 0.3, 0.4,
                0.1, 0.2, 0.3, 0.4,
                0.1, 0.2, 0.3, 0.4,
                0.1, 0.2, 0.3, 0.4,

                0.3, 0.2, 0.1, 0.4,
                0.3, 0.2, 0.1, 0.4,
                0.3, 0.2, 0.1, 0.4,
                0.3, 0.2, 0.1, 0.4,
                0.3, 0.2, 0.1, 0.4,

                0.9, 0.2, 0.1, 0.4,
                0.9, 0.2, 0.1, 0.4,
                0.9, 0.2, 0.1, 0.7,
                0.3, 0.2, 0.1, 0.7,
                0.3, 0.2, 0.1, 0.7,

                0.9, 0.2, 0.1, 0.4,
                0.9, 0.5, 0.6, 0.4,
                0.9, 0.5, 0.1, 0.7,
                0.3, 0.5, 0.1, 0.7,
                0.3, 0.2, 0.8, 0.7,
            };

            var att_res_data: [7*4*4]T = undefined;
            var com_vec_data = [5]T {0.1, -0.1, 0, 0.2, -0.2};
            pub var out_data: [5*7]T = undefined;

            pub var cont1 = [_]T {
                0,
                2,
                4,
            };
            pub var cont2 = [_]T {0, 1, 2};
            pub var cont3 = [_]T {
                0, 1,
                3, 4,
                6, 7,
            };
            var cont4 = [_]T {-1, -2, -3};
            var cont5 = [_]T {
                9, 8, 
                6, 5, 
                3, 2, 
                0,-1,
            };
            var cont6 = [_]T {2, 4, 6, 8};
            
            const atten = Attention(T) {
                    .header = header2.toAttentionHeader(),
                    .query_matrix = Matrix(T) {
                        .capacity = header2.att_dim*header2.seq_dim,
                        .height = header2.att_dim,
                        .width = header2.seq_dim,
                        .ptr = &cont1,
                    },

                    .query_vect = &cont2,

                    .key_matrix = Matrix(T) {
                        .capacity = header2.att_dim*header2.ctx_dim,
                        .height = header2.att_dim,
                        .width = header2.ctx_dim,
                        .ptr = &cont3,
                    },

                    .key_vect = &cont4,

                    .value_matrix = Matrix(T) {
                        .capacity = header2.out_dim*header2.ctx_dim,
                        .height = header2.mid_dim,
                        .width = header2.ctx_dim,
                        .ptr = &cont5,
                    },

                    .value_vect = &cont6,

                    .key = undefined,
                    .value = undefined,
                    .query = undefined,
                    .score = undefined,
                    .out = undefined,
            };

            var mid_data: [4][4*7]T = undefined;
            var attentions = [4]Attention(T) {atten, atten, atten, atten};

            pub const mhatt = MHAttention(T) {
                .header = header2,

                .attentions = &attentions,
                .att_results = Matrix(T) {
                    .capacity = 7*4*4,
                    .height = 7*4,
                    .width = 4,
                    .ptr = &att_res_data,
                },
                .comb_matrix = Matrix(T) {
                    .capacity = 5*4*4,
                    .height = 5*4,
                    .width = 4,
                    .ptr = &comb_mat_data,
                },
                .comb_vect = &com_vec_data,


                .out = Matrix(T){
                    .capacity = out_data.len,
                    .height = 7,
                    .width = 5,
                    .ptr = &out_data,
                },
            };
            
        };

        test allocateForHeader {
            const allocator = std.testing.allocator;
            var mhatt: Self = undefined;
            mhatt.header = testData.header;
            try mhatt.allocateForHeader(allocator);
            defer mhatt.destroy(allocator);
        }

        test init {
            const allocator = std.testing.allocator;
            var out = try Matrix(T).init(allocator, testData.header.max_seq_len,
                testData.header.out_dim);
            var mhatt = try init(allocator, testData.header, out);
            mhatt.destroy(allocator);
            out.destroy(allocator);
        }

        test fillRandom {
            const allocator = std.testing.allocator;
            var xoshiro = std.Random.Xoshiro256.init(123);
            const rand = xoshiro.random();
            var out = try Matrix(T).init(allocator, testData.header.max_seq_len,
                testData.header.out_dim);
            defer out.destroy(allocator);
            var mhatt = try init(allocator, testData.header, out);
            defer mhatt.destroy(allocator);

            mhatt.fillRandom(rand, 0.1);
        }

        test writeWeights {
            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .write_only});
            defer file.close();
            const writer = file.writer();

            var mhatt = testData.mhatt;
            for (mhatt.attentions, &testData.mid_data) |*a, *d| {
                a.out.ptr = d;
                a.out.capacity = 28;
            }

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
            try std.testing.expectEqualDeep(testData.header2, mhatt.header);
            try std.testing.expectEqualDeep(testData.header2.toAttentionHeader(), mhatt.attentions[2].header);

            // testing wights data
            try std.testing.expectEqualSlices(T, mhatt.attentions[0].query_matrix.toSlice(),
                &testData.cont1);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].query_vect,
                &testData.cont2);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].key_matrix.toSlice(),
                &testData.cont3);
            try std.testing.expectEqualSlices(T, mhatt.attentions[3].key_vect,
                &testData.cont4);
            try std.testing.expectEqualSlices(T, mhatt.attentions[2].value_matrix.toSlice(),
                &testData.cont5);
            try std.testing.expectEqualSlices(T, mhatt.attentions[1].value_vect,
                &testData.cont6);
            try std.testing.expectEqualSlices(T, mhatt.comb_matrix.toSlice(),
                &testData.comb_mat_data);
            try std.testing.expectEqualSlices(T, mhatt.comb_vect,
                &testData.com_vec_data);
        }
    };
}

comptime {
    _ = MHAttention(f32);
}
