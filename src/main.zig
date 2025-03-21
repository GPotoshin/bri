const std = @import("std");
const assert = std.debug.assert;

const tk = @import("tokenizer.zig");
const tr = @import("transformer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    const attention_file = std.fs.cwd().openFile("data/attention_base", .{.mode = .read_write}) catch |e| {
        std.debug.print("file 'data/tokens' is not found\n", .{});
        return e;
    };
    defer attention_file.close();
    
    // emb_dim = 768
    // att_dim = 768
    // ffl_dim = 3072
    // layers = 12
    // context = 1024
    // var attention = try tr.Attention(f32).init(allocator, .{.seq_dim = 768,
    //     .ctx_dim = 768, .attn_dim = 768, .out_dim = 768, .max_seq_len = 1024,
    //     .max_ctx_len = 1024});

    var b = std.Random.Xoroshiro128.init(12);
    const rand = b.random();

    const seq = try tr.Matrix(f32).init(allocator, 5, 768); 
    const ctx = try tr.Matrix(f32).init(allocator, 5, 768); 

    seq.fillRandom(rand, 0.001);
    ctx.fillRandom(rand, 0.001);

    const att = try tr.Attention(f32).initFromFile(allocator, attention_file);

    _ = try att.calculate(seq, ctx, .bidirectional);
    std.debug.print("{}\n", .{att.out.at_nocheck(1, 2)});

}
