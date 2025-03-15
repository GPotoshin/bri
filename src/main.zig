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

    const att = try tr.Attention(f32).initFromFile(allocator, attention_file);
    std.debug.print("{}, {}, {}, {}, {}, {}\n", .{att.seq_dim, att.ctx_dim,
    att.attn_dim, att.out_dim, att.max_seq_len, att.max_ctx_len});

}
