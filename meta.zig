const std = @import("std");

pub fn copyData(dst_ptr: anytype, src_ptr: anytype) void {
  @setEvalBranchQuota(1000_000);
  const src_type = std.meta.Child(@TypeOf(src_ptr));
  const dst_type = std.meta.Child(@TypeOf(dst_ptr));

  if (src_type == dst_type) {
    dst_ptr.* = src_ptr.*;
    return;
  } else if (dst_type == void) {
    return;
  }

  const src_typeinfo = @typeInfo(src_type);
  const dst_typeinfo = @typeInfo(dst_type);

  if (comptime std.meta.activeTag(src_typeinfo) != std.meta.activeTag(dst_typeinfo)) {
    @compileError("Cant copy `" ++ @typeName(src_type) ++ "` to `" ++ @typeName(dst_type) ++ "` as they are different types entirely");
  }

  switch (dst_typeinfo) {
    .@"struct" => |info| {
      inline for (info.fields) |field| {
        copyData(&@field(dst_ptr, field.name), &@field(src_ptr, field.name));
      }
    },
    .@"array" => |info| {
      inline for (0..info.len) |i| {
        copyData(&@field(dst_ptr, i), &@field(src_ptr, i));
      }
    },
    else => {
      @compileError("Cant copy `" ++ @typeName(src_type) ++ "` to `" ++ @typeName(dst_type) ++ "` as they are incompatible");
    },
  }
}

fn getStrRepr(T: type) []const u8 {
  comptime var retval: []const u8 = &.{};
  comptime {
    const typeinfo = @typeInfo(T);
    retval = retval ++ @typeName(T);

    switch (typeinfo) {
      .@"struct" => |info| {
        for (info.fields) |field| {
          retval = retval ++ field.name;
          retval = retval ++ getStrRepr(field.type);
        }
        for (info.decls) |decl| {
          retval = retval ++ decl.name;
          const dval = @field(T, decl.name);
          const dtype = @TypeOf(dval);
          if (dtype == type) {
            retval = retval ++ getStrRepr(dval);
          // } else {
          //   retval = retval ++ std.fmt.comptimePrint("{any}", .{dval});
          }
        }
      },
      .@"array" => |info| {
        retval = retval ++ getStrRepr(info.child);
        retval = retval ++ std.fmt.comptimePrint("[{d}]", .{info.len});
        if (info.sentinel()) |sentinel| {
          retval = retval ++ std.fmt.comptimePrint("{any}", .{sentinel});
        }
      },
      else => {},
    }
  }

  return retval;
}

pub fn hashType(T: type) u128 {
  comptime var hasher = std.hash.Fnv1a_128.init();
  comptime {
    const str_repr = getStrRepr(T);
    // @compileLog(str_repr);
    hasher.update(str_repr);
  }
  return comptime hasher.final();
}

