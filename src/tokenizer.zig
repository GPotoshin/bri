const std = @import("std");

const float_type = f16;

pub fn Token(comptime dim: u32) type {
    return struct {
        id: u32,
        vect: [dim]float_type,

        const Self = @This();
        pub fn random(id: u32) Self {
            return .{
                .id = id,
                .vect = undefined,
            };
        }
    };

}

pub fn Tokenizer(comptime dim: u32) type {
    return struct {
        tokens: std.StringArrayHashMap(Token(dim)),
        free_id: u32,

        const Self = @This();
        const max_token_len: u32 = 16;
        const ParsingError = error{
            DataConflict,
            WrongAlignement,
        };

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            // var buffer: [16]u8 = undefined; // tokens must be no longer then 16 chars
            var reader = file.reader();
            // The file should contain
            // ($dimension = 4 bytes)
            // $token_string $x_1 $x_2 ...
            // ...
            var tokens = std.StringArrayHashMap(Token(dim)).init(allocator);

            const dimension = try reader.readInt(u32, .little);
            if (dimension != dim) {
                std.debug.print("wrong dimension!\n", .{});
                return ParsingError.DataConflict;
            }

            var id: u32 = 0;
            while (reader.readBytesNoEof(max_token_len)) |string| {
                var slice = string[0..];
                const key = std.mem.trim(u8, slice, .{0});
                var token: Token(dim) = undefined; 
                slice.ptr = @ptrCast(&token.vect);
                slice.len = dim*@sizeOf(float_type);

                if (reader.readAll(slice) != slice.len) {
                    std.debug.print("wrong alignement, reader reached the end," ++
                                    " while vector did not end\n"
                                    , .{}); 
                    return ParsingError.WrongAlignement;
                }

                token.id = id;
                id += 1;
                tokens.put(key, token);
            }
            
            return .{
                .tokens = tokens,
            };
        }

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .tokens = std.StringArrayHashMap(Token(dim)).init(allocator),
                .free_id = 0,
            };
        }

        fn find_prefix_token (self: *Self, slice: []u8) ?Token(dim) {
            var slice_copy = slice;
            while (slice_copy.len > 0) {
                if (self.tokens.get(slice_copy)) |token| {
                    return token;
                }
                slice_copy.len -= 1;
            }
            return null;
        }

        pub fn tokenize(self: *Self, allocator: std.mem.Allocator, file: std.fs.File) !void {
            var reader = file.reader();
            var buffer: [128]u8 = undefined;
            var list = std.ArrayList([dim]float_type).init(allocator);
            while (reader.readUntilDelimiter(buffer[0..], ' ')) |slice| {
                if (self.find_prefix_token(slice)) |token| {
                    try list.append(token.vect);
                } else {
                    const stdout = std.io.getStdOut().writer();
                    const stdin = std.io.getStdIn().reader();

                    try stdout.print("Enter new tokens for '{s}' in correct" ++
                        "order, write 'Done.' to end\n" , .{slice});
                    while (true) {
                        var input: [128]u8 = undefined;
                        const buf = stdin.readUntilDelimiter(input[0..], '\n') catch {
                            try stdout.print("the string is to long\n", .{});
                            continue;
                        };
                        if (std.mem.eql(u8, buf, "Done.")) break;
                        const new_token = Token(dim).random(self.free_id);
                        self.free_id += 1;

                        try self.tokens.put(buf, new_token);
                        try list.append(new_token.vect);
                    }
                }
            } else |e| {
                std.debug.print("The Word is to long\n", .{});
                return e;
            }
        }

        pub fn writeToFile(self: *Self, file: std.fs.File) !void {
            var bytes: []u8 = undefined;
            var dimension = dim;
            bytes.ptr = @ptrCast(&dimension);
            bytes.len = 4;

            const writer = file.writer();
            writer.writeAll(bytes) catch |e| {
                std.debug.print("Could not write\n", .{}); 
                return e;
            };
            var t_iter = self.tokens.iterator();
            while (t_iter.next()) |el| {
                try writer.writeAll(el.key_ptr.*);
                try writer.writeByteNTimes(0, max_token_len-el.key_ptr.len);

                bytes.len = @sizeOf(@TypeOf(el.value_ptr.vect));
                bytes.ptr = @ptrCast(&el.value_ptr.vect);
                try writer.writeAll(bytes);
            }
        }
    };
}

test "len of vector" {
    std.debug.print("len of [8]u8 is {}\n", .{@sizeOf([8]u8)});
}
