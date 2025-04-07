const std = @import("std");

const debug = true;
const allowed = .{
  .ok = false,
};

fn isAllowed(comptime src: *const std.builtin.SourceLocation) bool {
  inline for (@typeInfo(@TypeOf(allowed)).@"struct".fields) |field| {
    if (comptime !std.mem.eql(u8, field.name, src.file)) continue;
    const val = @field(allowed, field.name);
    if (@TypeOf(val) == bool) return val;

    inline for (@typeInfo(@TypeOf(val)).@"struct".fields) |field2| {
      if (comptime !std.mem.eql(u8, field2.name, src.fn_name)) continue;
      return @field(val, field2.name);
    }
  }
  return false;
}

var gpa: if (debug) std.heap.GeneralPurposeAllocator(.{}) else void = .{};
var donemap: if (debug) std.AutoHashMap(usize, void) else void = undefined;
pub fn init() void {
  if (debug) {
    donemap = @TypeOf(donemap).init(gpa.allocator());
  }
}

const stdout = std.io.getStdOut().writer();
var buffered_stdout = std.io.bufferedWriter(stdout);
const writer = buffered_stdout.writer();

pub fn log(comptime src: *const std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
  if (debug) {
    if (!isAllowed(src)) {
      const gp = donemap.getOrPut(@intFromPtr(src)) catch return;
      if (gp.found_existing) return;
      writer.print("!! {any}\n", .{src}) catch {};
    } else {
      writer.print(">> {any}\n", .{src}) catch {};
      writer.print(format, args) catch {};
    }
  } else {
    if (!isAllowed(src)) return;
    writer.print(format, args) catch {};
  }
}

pub fn flushlog(comptime src: *const std.builtin.SourceLocation) void {
  if (!isAllowed(src)) return;
  buffered_stdout.flush() catch {};
}

pub fn logflushing(comptime src: *const std.builtin.SourceLocation, comptime format: []const u8, args: anytype) void {
  if (!isAllowed(src)) return;
  writer.print(format, args) catch {};
  flushlog(src);
}

