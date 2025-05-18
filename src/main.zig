const std = @import("std");
const assert = std.debug.assert;

const tk = @import("tokenizer.zig");
const mtx = @import("matrix.zig");
const ed = @import("edtransformer.zig");

const Matrix = mtx.Matrix;
const EDTransformer = ed.EDTransformer;
const EDHeader = ed.EDHeader;

const TrainingData = struct {
    ctx: []u8,
    seq: []u8,
};


pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    var xoshiro = std.Random.Xoshiro256.init(321);
    const rand = xoshiro.random();

    var args = std.process.args();
    assert(args.skip());
    
    const mode = args.next() orelse {
        std.debug.print("Please follow executable with mode (tokenize | train)\n", .{});
        return;
    };

    const tokens_file = std.fs.cwd().openFile("data/tokens", .{.mode = .read_write}) catch |e| {
        std.debug.print("file 'data/tokens' is not found\n", .{});
        return e;
    };
    defer tokens_file.close();
    var tokenizer = try tk.Tokenizer().initFromFile(allocator, tokens_file);

    if (std.mem.eql(u8, mode, "tokenize")) {
        const text_file_name_opt = args.next();
        const text_file_name = text_file_name_opt orelse {
            std.debug.print("You should pass a text file name as a first argument\n", .{});
            return;
        };

        var file = std.fs.cwd().openFileZ(text_file_name, .{.mode = .read_only}) catch |e| {
            std.debug.print("file is not found", .{});
            return e;
        }; 
        defer file.close();

        const file_reader = file.reader();
        _ = try tokenizer.tokenize(allocator, file_reader);
    } else if (std.mem.eql(u8, mode, "train")) {

        const payload_file = try std.fs.cwd().openFile("library/ctx_and_seq.json", .{.mode = .read_only});
        const json_payload = try payload_file.readToEndAlloc(allocator, 10000);
        const payload = try std.json.parseFromSliceLeaky(TrainingData, allocator, json_payload, .{});

        var ctx_stream = std.io.fixedBufferStream(payload.ctx);
        const ctx_reader = ctx_stream.reader();
        const ctx_ids = try tokenizer.tokenize(allocator, ctx_reader);

        var seq_stream = std.io.fixedBufferStream(payload.seq);
        const seq_reader = seq_stream.reader();
        const seq_ids = try tokenizer.tokenize(allocator, seq_reader);
        
        try tokenizer.writeToFile(tokens_file);

        const embedding_file = std.fs.cwd().openFile("data/embedding", .{.mode = .read_write}) catch |e| {
            std.debug.print("file 'data/embedding' is not found\n", .{});
            return e;
        };
        defer embedding_file.close();
        var embedding = try mtx.Matrix(f32).initFromFile(allocator, embedding_file);
        if (embedding.height < tokenizer.tokens.items.len) {
            if (!allocator.resize(embedding.ptr, embedding.width*tokenizer.tokens.items.len)) {
                std.debug.print("Could not resize embedding\n", .{});
                return;
            }
            embedding.ptr.len = embedding.width*tokenizer.tokens.items.len;
            const old_height = embedding.height;
            embedding.height = tokenizer.tokens.items.len;
            embedding.submatrix(old_height, embedding.height).?.fillRandom(rand, 1);
        } else if (embedding.height > tokenizer.tokens.items.len) {
            std.debug.print("we cannot delete embeddings yet\n", .{});
            return;
        }
        try embedding.writeToFile(embedding_file);

        const ctx_mat = try Matrix(f32).init(allocator, ctx_ids.items.len, embedding.width);
        for (0..ctx_ids.items.len) |i| {
            const row = ctx_mat.row(i);
            std.mem.copyBackwards(f32, row, embedding.row(ctx_ids.items[i])); // we have a bug here
            @import("position.zig").addPositionInformation(row, i, 10000); 
        }

        const seq_mat = try Matrix(f32).init(allocator, seq_ids.items.len, embedding.width);
        for (0..seq_ids.items.len) |i| {
            const row = seq_mat.row(i);
            std.mem.copyBackwards(f32, row, embedding.row(seq_ids.items[i])); // we have a bug here
            @import("position.zig").addPositionInformation(row, i, 10000); 
        }

        const transformer_file = std.fs.cwd().openFile("data/edtransformer_0.0.0", .{.mode = .read_write }) catch |e| {
            std.debug.print("file 'data/edtransformer_0.0.0' is not found\n", .{});
            return e;
        };
        defer transformer_file.close();
        var transformer = try EDTransformer(f32).initFromReader(allocator, transformer_file.reader(), embedding);
        try transformer.compute(ctx_mat, seq_mat);
        std.debug.print("height: {}, width: {}, seq_height: {}\n", .{transformer.out.height,
            transformer.out.width, seq_mat.height});

        try transformer.writeToFile(transformer_file);
        try embedding.writeToFile(embedding_file);
    }
    try tokenizer.writeToFile(tokens_file);
}
