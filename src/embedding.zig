const std = @import("std");
const parsing = @import("parsing.zig");

const mtx = @import("matrix.zig");
const Matrix = mtx.Matrix;

pub fn readEmdedingFile(comptime T: type, allocator: std.mem.Allocator, file: std.fs.File) !Matrix(T) {
    const reader = file.reader();

    const typelen = try reader.readInt(u32, .little) catch |e| {
        std.log.err("Can't read typelen from an embending file\n", .{});
        return e;
    };
    if (typelen != @sizeOf(T)) {
        std.log.err("embeding file contains a different type\n", .{});
        return error.DataConflict;
    }
    const edim = try reader.readInt(u32, .little) catch |e| {
        std.log.err("Can't read embeding dimension from an embeding file\n", .{});
        return e;
    };
    const entries_number = try reader.readInt(u32, .little) catch |e| {
        std.log.err("Can't read entries number from an embending file\n", .{});
        return e;
    };
    
    const entries = try Matrix(T).init(allocator, entries_number, edim);
    try entries.read(reader);
}

pub fn writeEmdedingFile(comptime T: type, embedding: Matrix(T), file: std.fs.File) !Matrix(T) {
    const writer = file.writer();

    writer.writeInt(u32, @sizeOf(T), .little) catch |e| {
        std.log.err("Can't write typelen to an embending file\n", .{});
        return e;
    };
    writer.writeInt(u32, embedding.width, .little) catch |e| {
        std.log.err("Can't write embeding dimension to an embeding file\n", .{});
        return e;
    };
    writer.writeInt(u32, embedding.height, .little) catch |e| {
        std.log.err("Can't write entries number to an embending file\n", .{});
        return e;
    };
    
    try embedding.write(writer);
}
