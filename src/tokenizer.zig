const std = @import("std");
const parsing = @import("parsing.zig");


pub fn Tokenizer() type {
    return struct {
        ids: std.StringHashMap(u32),
        tokens: std.ArrayList([]u8),
        free_id: u32,

        const Self = @This();
        const max_token_len = 16;

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


        /// reads from a file tokens with their IDs. IDs should coincide with
        /// tokens position. It's used to insure the correctness of algorithms
        pub fn initFromFile(allocator: std.mem.Allocator, file: std.fs.File) !Self {
            var reader = file.reader();
            // The file should contain
            // $number_of_tokens
            // $token_string[16 bytes] $token_id[4 bytes]
            // ...
            // as tokens are supposed to be ordered by ids their aim is to show
            // that the data is coherent
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

            var tokens = try std.ArrayList([]u8).initCapacity(allocator, tokens_stored);
            const handle = tokens.addManyAsSliceAssumeCapacity(tokens_stored);
            
            var tokens_read_counter: u32 = 0;

            while (reader.readBytesNoEof(max_token_len)) |string| {
                const slice = trim(@constCast(string[0..]), 0);
                const key = try allocator.alloc(u8, slice.len);
                std.mem.copyForwards(u8, key, slice);

                const id = reader.readInt(u32, .little) catch |e| {
                    std.debug.print("Could not read tokens id\n", .{});
                    return e;
                };

                if (id >= tokens_stored) {
                    std.debug.print("ID is out of bound\n", .{});
                    return error.DataConflict;
                }

                if (tokens_read_counter != id) {
                    std.debug.print("Wrong ID: {} at place {}\n", .{id, tokens_read_counter});
                    return error.DataConflict;
                }

                try ids.put(key, id);
                handle[id] = key;

                tokens_read_counter += 1;
            } else |e| {
                if (e == error.EndOfStream) {
                    if (tokens_read_counter != tokens_stored) {
                        std.debug.print("The file is corupted\n", .{});
                        return error.DataConflict;
                    }
                    return .{
                    .ids = ids,
                    .tokens = tokens,
                    .free_id = tokens_stored,
                    };
                }
                std.debug.print("Can't read token\n", .{});
                return e;
            }
            
            return .{
                .ids = ids,
                .tokens = tokens,
                .free_id = tokens_stored,
            };
        }

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .ids = std.StringArrayHashMap(u32).init(allocator),
                .tokens = std.ArrayList(u32),
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
        fn find_prefix_token (self: *Self, slice: []u8) !?struct{ id: u32, l: u32} {
            const stdout = std.io.getStdOut().writer();
            var slice_copy = slice;
            if (not_exception(slice)) {
                while (slice_copy.len > 0) {
                    if (self.ids.get(slice_copy)) |id| {
                        try stdout.print("{s}|", .{slice_copy});
                        return .{.id = id, .l = @truncate(slice_copy.len)};
                    }
                    slice_copy.len -= 1;
                }
            }
            slice_copy = slice;
            if (slice.len >= 2 and 'A' <= slice[1] and slice[1] <= 'Z') {
                slice[1] += 32; //'a' - 'A'
                while (slice_copy.len > 0) {
                    if (self.ids.get(slice_copy)) |id| {
                        try stdout.print("{s}|", .{slice_copy});
                        return .{.id = id, .l = @truncate(slice_copy.len)};
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

        /// Stores token and associates to it self.free_id and increaments it
        fn store_new_token(self: *Self, token: []u8) !void {
            try self.ids.put(token, self.free_id);
            try self.tokens.insert(self.free_id, token);
            self.free_id += 1;
        }

        /// Removes ids and restores data. If there is a model associated to
        /// this data, it also changes it's indicies. There are no models for
        /// now but in the future we will need to add some sort of callback
        fn remove_id(self: *Self, id: u32) !void {
            const key = self.tokens.swapRemove(id);
            try self.ids.put(key, id);
        }

        fn tokenize_word(self: *Self, allocator: std.mem.Allocator, slice: []u8,
            list: *std.ArrayList(u32)) !void {

            var mut_slice = @constCast(slice);
            const full_len = mut_slice.len;
            var sum_len: u32 = 0;

            const stdout = std.io.getStdOut().writer();
            const stdin = std.io.getStdIn().reader();


            var token_count: u32 = 0;
            while (sum_len < full_len) {
                if (try self.find_prefix_token(mut_slice)) |token| {
                    try list.append(token.id);
                    token_count += 1;
                    sum_len += token.l;
                    mut_slice.ptr += token.l;
                    mut_slice.len -= token.l;
                } else {
                    // try stdout.print("Unknown sequence {any}\n", .{mut_slice});
                    try stdout.print("\nEnter new prefix token for '{s}'\n" , .{mut_slice});
                    var input: [128]u8 = undefined;
                    const buf = stdin.readUntilDelimiter(input[0..], '\n') catch {
                        try stdout.print("the string is to long\n", .{});
                        continue;
                    };

                    if (buf.len == 0) {
                        if (mut_slice.len > max_token_len) {
                            try stdout.print("the token is too long, it should"
                                ++ " be shoreter then {} bytes\n", .{max_token_len});
                            continue;
                        }
                        const key = try allocator.alloc(u8, mut_slice.len);
                        std.mem.copyForwards(u8, key, mut_slice);
                        try list.append(self.free_id); // it's imporatant to do it before store_new_token
                        try self.store_new_token(key);
                        token_count += 1;
                        sum_len += @truncate(mut_slice.len);
                    }
                    else if (std.mem.eql(u8, buf, "Restart.")){
                        sum_len = 0;
                        mut_slice = slice;
                        for (0..token_count) |_| {
                            _ = list.pop();
                        }
                        token_count = 0;
                    }
                    else if (buf.len >= 4 and std.mem.eql(u8, buf[0..4], "Add.")) {
                        const new_string = buf[4..];
                        if (new_string.len > max_token_len) {
                            try stdout.print("the token is too long, it should"
                                ++ " be shoreter then {} bytes\n", .{max_token_len});
                            continue;
                        }
                        const key = try allocator.alloc(u8, new_string.len);
                        std.mem.copyForwards(u8, key, new_string);

                        try self.store_new_token(key);
                    }
                    else if (buf.len >= 7 and std.mem.eql(u8, buf[0..7], "Remove.")) {
                        if (self.ids.get(buf[7..])) |id| {
                            _ = self.ids.remove(buf[7..]);
                            try self.remove_id(id);
                        }
                    }
                    else if (buf.len >= 5 and std.mem.eql(u8, buf[0..5], "Find.")) {
                        var iterator = self.ids.iterator();
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
                        try self.store_new_token(key);
                        token_count += 1;

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
            list: *std.ArrayList(u32)) !void {
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

        // the function takes a file (remplace for a reader?) and outputs a
        // list of token indices. The list is allocated with a given allocator
        pub fn tokenize(self: *Self, allocator: std.mem.Allocator, file: std.fs.File) !std.ArrayList(u32) {
            const stdout = std.io.getStdOut().writer();
            var reader = file.reader();
            var buffer: [128]u8 = undefined;
            var list = std.ArrayList(u32).init(allocator);
            var fbs = std.io.fixedBufferStream(buffer[1..]);
            while (reader.streamUntilDelimiter(fbs.writer(), ' ', fbs.buffer.len)) {
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
                    try stdout.print("\n", .{});
                    return list;
                }
                std.debug.print("The Word is to long\n", .{});
                return e;
            }
            try stdout.print("\n", .{});
            return list;
        }

        pub fn write(self: Self, writer: anytype) !void {
            const elnum = self.tokens.items.len;
            try writer.writeInt(u32, @truncate(elnum), .little);

            for (self.tokens.items, 0..) |token, i| {
                try writer.writeAll(token);
                try writer.writeByteNTimes(0, max_token_len - token.len);
                try writer.writeInt(u32, @truncate(i), .little);
            }
        }

        pub fn writeToFile(self: Self, file: std.fs.File) !void {
            try file.seekTo(0);

            const writer = file.writer();
            self.write(writer);

            try file.setEndPos(try file.getPos());
        }
    };
}

test "little" {
    const file = (try std.fs.cwd().openFile("test", .{.mode = .write_only})).writer();
    try file.writeInt(u32, 428, .little);
    
}
