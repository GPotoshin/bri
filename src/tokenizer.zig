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

pub fn Tokenizer() type {
    return struct {
        ids: std.StringHashMap(u32),
        tokens: std.ArrayList([]u8),
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
            var reader = file.reader();
            // The file should contain
            // $number_of_tokens
            // $token_string[16 bytes] $token_id[4 bytes]
            // ...
            var ids = std.StringHashMap(u32).init(allocator); // it's arena

            const endPos = try file.getEndPos();
            if (endPos == 0) {
                std.debug.print("empty file...\n", .{});
                return .{
                    .tokens = std.ArrayList([]u8).init(allocator),
                    .ids = ids,
                    .free_id = 0,
                };
            }

            const tokens_stored = reader.readInt(u32, .little) catch |e| {
                std.debug.print("could not read first 4 bytes from tokens file\n", .{});
                return e;
            };

            var tokens = std.ArrayList([]u8).initCapacity(allocator, tokens_stored);
            
            var tokens_read_counter = 0;

            while (reader.readBytesNoEof(max_token_len)) |string| {
                const slice = trim(@constCast(string[0..]), 0);
                const key = try allocator.alloc(u8, slice.len);
                std.mem.copyForwards(u8, key, slice);

                const id = reader.readInt(u32, .little) catch |e| {
                    std.debug.print("Could not read tokens id\n", .{});
                    return e;
                };

                try ids.put(key, id);
                try tokens.insert(id, key);

                tokens_read_counter += 1;
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
            if (slice.len >= 3 and slice[1] == 'V' and 'a' <= slice[2] and
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

        fn not_in_list(el: u8, list: []u8) bool {
            for (list) |a| {
                if (a == el) {
                    return false;
                }
            }
            return true;
        }

        fn split_and_call(self: *Self, allocator: std.mem.Allocator, buffer: *[128]u8, len: u32,
            list: *std.ArrayList([dim]float_type)) !void {
            var start: u32 = 0;
            var end: u32 = 1;
            while (end+1 < len) {
                buffer[start] = ' ';
                var separate = [_]u8{'\n', '"', '\'', '('};
                while (end+1 < len and not_in_list(buffer[end], separate[0..])) {
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

        pub fn writeTokens(self: *Self, file: std.fs.File) !void {
            try file.seekTo(0);

            const writer = file.writer();

            var t_iter = self.tokens.iterator();
            while (t_iter.next()) |el| {
                try writer.writeAll(el.key_ptr.*);
                try writer.writeByteNTimes(0, max_token_len-el.key_ptr.len);
                try writer.writeInt(u32, el.value_ptr, .little);
            }
        }
    };
}
