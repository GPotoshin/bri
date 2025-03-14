const std = @import("std");
const assert = std.debug.assert;

const tk = @import("tokenizer.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    var args = std.process.args();
    assert(args.skip());

    const filename_opt = args.next();
    const filename = filename_opt orelse {
        std.debug.print("You should pass a file as a first argument\n", .{});
        return;
    };

    var file = std.fs.cwd().openFileZ(filename, .{.mode = .read_only}) catch |e| {
        std.debug.print("file is not found", .{});
        return e;
    }; 
    defer file.close();

    const tokens_file = std.fs.cwd().openFile("data/tokens", .{.mode = .read_write}) catch |e| {
        std.debug.print("file 'data/tokens' is not found\n", .{});
        return e;
    };
    defer tokens_file.close();

    var tokenizer = try tk.Tokenizer().initFromFile(allocator, tokens_file);

    _ = try tokenizer.tokenize(allocator, file);
    try tokenizer.writeToFile(tokens_file);
}
