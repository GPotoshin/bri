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

        // @Tests
        const testData = struct {
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

        test initFromFile {
            const allocator = std.testing.allocator;

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .read_only});
            defer file.close();
            var mhatt = try MHAttention(T).initFromFile(allocator, file);
            defer mhatt.destroy(allocator);


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

        test compute {
            const allocator = std.testing.allocator;

            const file = try std.fs.cwd().openFile("test_files/test_mhattention", .{.mode = .read_only});
            defer file.close();
            var mhatt = try MHAttention(T).initFromFile(allocator, file);
            defer mhatt.destroy(allocator);
            var out = try Matrix(T).init(allocator, mhatt.header.max_seq_len,
                mhatt.header.out_dim);
            defer out.destroy(allocator);
            mhatt.out = out;

            var seq_data = [_]T { 0.1, 0.3, 0.2, 0.4 };
            var ctx_data = [_]T { 0.1, 0.3, 0.2, 0.4, 0.3, 0.4, 0.0, -0.1 };

            const seq = Matrix(T) {
                .capacity = 4,
                .height = 4,
                .width = 1,
                .ptr = &seq_data,
            };
            const ctx = Matrix(T) {
                .capacity = 8,
                .height = 4,
                .width = 2,
                .ptr = &ctx_data,
            };

            try mhatt.compute(ctx, seq, .bidirectional);

            // right results
            try std.testing.expect(@abs(mhatt.out.toSlice()[0]-39.63546734) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[1]-45.51779338) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[2]-46.37794993) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[3]-37.45404993) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[4]-40.09511809) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[5]-39.9070572) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[6]-45.82308028) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[7]-46.6671428) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[8]-37.6305832) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[9]-40.27416612) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[10]-39.78800458) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[11]-45.68924128) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[12]-46.54031778) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[13]-37.55304782) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[14]-40.19549336) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[15]-40.00224998) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[16]-45.93011503) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[17]-46.7686183) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[18]-37.69276302) < 0.00001);
            try std.testing.expect(@abs(mhatt.out.toSlice()[19]-40.33729753) < 0.00001);
        }
    };
}

comptime {
    _ = MHAttention(f32);
}
