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
        tokens: std.StringHashMap(Token(dim)),
        free_id: u32,

        const Self = @This();
        const max_token_len: u32 = 16;
        const ParsingError = error{
            DataConflict,
            WrongAlignement,
        };

        fn trim(slice: []u8, char: u8) []u8 {
            var retval: []u8 = undefined;
            retval.ptr = slice.ptr;
            retval.len = 0;
            while (retval.len < slice.len) {
                if (retval.ptr[retval.len] == char) {
                    break;
                }
                retval.len += 1;
            }

            return retval;
        }

        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            // var buffer: [16]u8 = undefined; // tokens must be no longer then 16 chars
            var reader = file.reader();
            // The file should contain
            // ($dimension = 4 bytes)
            // $token_string $x_1 $x_2 ...
            // ...
            var tokens = std.StringHashMap(Token(dim)).init(allocator);
            const endPos = try file.getEndPos();
            if (endPos == 0) {
                std.debug.print("empty file...\n", .{});
                return .{
                    .tokens = tokens,
                    .free_id = 0,
                };
            }

            const dimension = try reader.readInt(u32, .little);
            if (dimension != dim) {
                std.debug.print("wrong dimension!\n", .{});
                return ParsingError.DataConflict;
            }

            var id: u32 = 0;
            while (reader.readBytesNoEof(max_token_len)) |string| {
                var slice = trim(@constCast(string[0..]), 0);
                const key = try allocator.alloc(u8, slice.len);
                std.mem.copyForwards(u8, key, slice);
                var token: Token(dim) = undefined; 
                slice.ptr = @ptrCast(&token.vect);
                slice.len = dim*@sizeOf(float_type);

                if (try reader.readAll(slice) != slice.len) {
                    std.debug.print("wrong alignement, reader reached the end," ++
                                    " while vector did not end\n"
                                    , .{}); 
                    return ParsingError.WrongAlignement;
                }

                token.id = id;
                id += 1;
                try tokens.put(key, token);
            } else |e| {
                if (e == error.EndOfStream)  return .{
                    .tokens = tokens,
                    .free_id = id,
                };
                std.debug.print("Can't read token\n", .{});
                return e;
            }
            
            return .{
                .tokens = tokens,
                .free_id = id,
            };
        }

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .tokens = std.StringArrayHashMap(Token(dim)).init(allocator),
                .free_id = 0,
            };
        }

        // This code is highly coupled with the token base and the fact that
        // all words except proper nouns are read miniscule
        fn not_exception(slice: []u8) bool {
            if (slice.len >= 3 and slice[1] == 'I' and 'a' <= slice[2] and
                slice[2] <= 'z') {
                return false;
            }
            return true;
        }

        // Tokens are found in lower case if possible
        fn find_prefix_token (self: *Self, slice: []u8) !?struct{ t: Token(dim), l: u32} {
            const stdout = std.io.getStdOut().writer();
            var slice_copy = slice;
            if (not_exception(slice)) {
                while (slice_copy.len > 0) {
                    if (self.tokens.get(slice_copy)) |token| {
                        try stdout.print("{s}|", .{slice_copy});
                        return .{.t = token, .l = @truncate(slice_copy.len)};
                    }
                    slice_copy.len -= 1;
                }
            }
            slice_copy = slice;
            if (slice.len >= 2 and 'A' <= slice[1] and slice[1] <= 'Z') {
                slice[1] += 32; //'a' - 'A'
                while (slice_copy.len > 0) {
                    if (self.tokens.get(slice_copy)) |token| {
                        try stdout.print("{s}|", .{slice_copy});
                        return .{.t = token, .l = @truncate(slice_copy.len)};
                    }
                    slice_copy.len -= 1;
                }

                slice[1] -= 32;
            }
            return null;
        }

        const EndError = error{
            TokenizationAborted,
        };

        fn tokenize_word(self: *Self, allocator: std.mem.Allocator, slice: []u8,
            list: *std.ArrayList([dim]float_type)) !void {

            var mut_slice = @constCast(slice);
            const full_len = mut_slice.len;
            var sum_len: u32 = 0;

            var stored_token_count: u32 = 0;
            while (sum_len < full_len) {
                if (try self.find_prefix_token(mut_slice)) |token| {
                    try list.append(token.t.vect);
                    stored_token_count += 1;
                    sum_len += token.l;
                    mut_slice.ptr += token.l;
                    mut_slice.len -= token.l;
                } else {
                    const stdout = std.io.getStdOut().writer();
                    const stdin = std.io.getStdIn().reader();

                    // try stdout.print("Unknown sequence {any}\n", .{mut_slice});
                    try stdout.print("\nEnter new prefix token for '{s}'\n" , .{mut_slice});
                    var input: [128]u8 = undefined;
                    const buf = stdin.readUntilDelimiter(input[0..], '\n') catch {
                        try stdout.print("the string is to long\n", .{});
                        continue;
                    };

                    const new_token = Token(dim).random(self.free_id);
                    self.free_id += 1;

                    if (buf.len == 0) {
                        const key = try allocator.alloc(u8, mut_slice.len);
                        std.mem.copyForwards(u8, key, mut_slice);
                        try self.tokens.put(key, new_token);
                        try list.append(new_token.vect);
                        stored_token_count += 1;
                        sum_len += @truncate(mut_slice.len);
                    }
                    else if (std.mem.eql(u8, buf, "Restart.")){
                        sum_len = 0;
                        mut_slice = slice;
                        for (0..stored_token_count) |_| {
                            _ = list.pop();
                        }
                        stored_token_count = 0;
                    }
                    else if (buf.len >= 4 and std.mem.eql(u8, buf[0..4], "Add.")) {
                        const new_string = buf[4..];
                        const key = try allocator.alloc(u8, new_string.len);
                        std.mem.copyForwards(u8, key, new_string);

                        try self.tokens.put(key, new_token);
                    }
                    else if (buf.len >= 7 and std.mem.eql(u8, buf[0..7], "Remove.")) {
                        if (!self.tokens.remove(buf[7..])) {
                            try stdout.print("Didn't find!\n", .{});
                        }
                    }
                    else if (buf.len >= 5 and std.mem.eql(u8, buf[0..5], "Find.")) {
                        var iterator = self.tokens.iterator();
                        while (iterator.next()) |el| {
                            if (el.key_ptr.len >= buf.len-5 and std.mem.eql(u8,
                                    buf[5..], el.key_ptr.*[0..(buf.len-5)])) {
                                try stdout.print("token: {s}\n", .{el.key_ptr.*});
                            }
                        }
                    }
                    else if (std.mem.eql(u8, buf, "Exit.")) {
                        return EndError.TokenizationAborted;
                    }

                    else if (buf.len <= mut_slice.len and std.mem.eql(u8, buf, mut_slice[0..buf.len])) {
                        const key = try allocator.alloc(u8, buf.len);
                        std.mem.copyForwards(u8, key, buf);

                        try self.tokens.put(key, new_token);
                        try list.append(new_token.vect);
                        stored_token_count += 1;

                        sum_len += @truncate(buf.len);
                        mut_slice.ptr += buf.len;
                        mut_slice.len -= buf.len;
                    } else {
                        try stdout.print("Incorrect prefix\n", .{});
                    }
                }
            }
        }

        fn split_and_call(self: *Self, allocator: std.mem.Allocator, buffer: *[128]u8, len: u32,
            list: *std.ArrayList([dim]float_type)) !void {
            var start: u32 = 0;
            var end: u32 = 1;
            while (end+1 < len) {
                buffer[start] = ' ';
                while (end+1 < len and buffer[end] != '\n') {
                    end += 1;
                }
                try self.tokenize_word(allocator, buffer[start..(end+1)], list);
                start = end;
                end += 1;
            }
        }

        pub fn tokenize(self: *Self, allocator: std.mem.Allocator, file: std.fs.File) !std.ArrayList([dim]float_type) {
            var reader = file.reader();
            var buffer: [128]u8 = undefined;
            var list = std.ArrayList([dim]float_type).init(allocator);
            var fbs = std.io.fixedBufferStream(buffer[1..]);
            while (reader.streamUntilDelimiter(fbs.writer(), ' ', fbs.buffer.len)) { // What if there is a new line
                self.split_and_call(allocator, &buffer, @truncate(fbs.pos+1), &list) catch |e| {
                    if (e == error.TokenizationAborted) return list;
                    return e;
                };
                fbs.reset();
            } else |e| {
                if (e == error.EndOfStream) {
                    self.split_and_call(allocator, &buffer, @truncate(fbs.pos+1), &list) catch |e1| {
                        if (e1 != error.TokenizationAborted) return e1;
                    };
                    return list;
                }
                std.debug.print("The Word is to long\n", .{});
                return e;
            }
            return list;
        }

        pub fn writeToFile(self: *Self, file: std.fs.File) !void {
            try file.seekTo(0);
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
