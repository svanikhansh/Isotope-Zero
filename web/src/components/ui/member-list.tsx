"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface Member {
  id: string;
  name: string;
  initials: string;
  avatar?: string;
  color?: string;
  role?: string;
  status?: "online" | "away" | "offline" | "busy";
}

export interface MemberListProps {
  members: Member[];
  maxVisible?: number;
  className?: string;
  size?: "sm" | "md" | "lg";
  showStatus?: boolean;
  onMemberClick?: (member: Member) => void;
}

const sizeClasses = {
  sm: "h-7 w-7 text-[10px]",
  md: "h-9 w-9 text-xs",
  lg: "h-11 w-11 text-sm",
};

const statusSizeClasses = {
  sm: "h-2 w-2",
  md: "h-2.5 w-2.5",
  lg: "h-3 w-3",
};

const statusColors = {
  online: "bg-[var(--success)]",
  away: "bg-[var(--warning)]",
  busy: "bg-[var(--error)]",
  offline: "bg-[var(--moss)]",
};

function getInitials(name: string): string {
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
}

function getColorFromName(name: string): string {
  const colors = [
    "#90CAF9", "#4FC3F7", "#29B6F6", "#26C6DA",
    "#5C6BC0", "#7E57C2", "#42A5F5", "#1565C0",
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

export function MemberList({
  members,
  maxVisible = 5,
  className,
  size = "md",
  showStatus = true,
  onMemberClick,
}: MemberListProps) {
  const visibleMembers = members.slice(0, maxVisible);
  const remainingCount = members.length - maxVisible;

  return (
    <div className={cn("flex items-center", className)} role="list" aria-label="Team members">
      {visibleMembers.map((member, index) => (
        <motion.div
          key={member.id}
          initial={{ opacity: 0, x: -10, scale: 0.9 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          transition={{ delay: index * 0.05 }}
          className={cn(
            "relative flex-shrink-0",
            index > 0 && "-ml-3",
            onMemberClick && "cursor-pointer hover:z-10"
          )}
          onClick={() => onMemberClick?.(member)}
          role="listitem"
        >
          <div className="relative">
            <div
              className={cn(
                "rounded-full border-2 border-[var(--background)] overflow-hidden",
                "bg-[var(--muted)]",
                sizeClasses[size],
                "flex items-center justify-center font-medium select-none",
                "transition-transform duration-200 hover:scale-110 hover:z-20"
              )}
              style={{
                color: member.color || getColorFromName(member.name),
                backgroundColor: member.avatar ? "transparent" : `${member.color || getColorFromName(member.name)}20`,
              }}
            >
              {member.avatar ? (
                <img
                  src={member.avatar}
                  alt={member.name}
                  className="h-full w-full object-cover"
                />
              ) : (
                member.initials || getInitials(member.name)
              )}
            </div>

            {showStatus && member.status && (
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className={cn(
                  "absolute bottom-0 right-0 rounded-full border-2 border-[var(--background)]",
                  statusColors[member.status || "offline"],
                  statusSizeClasses[size]
                )}
              />
            )}
          </div>
        </motion.div>
      ))}

      {remainingCount > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className={cn(
            "flex-shrink-0 -ml-3",
            "rounded-full border-2 border-[var(--background)]",
            "bg-[var(--muted)] text-[var(--moss)] font-medium",
            "flex items-center justify-center select-none",
            sizeClasses[size]
          )}
          role="listitem"
          aria-label={`${remainingCount} more members`}
        >
          +{remainingCount}
        </motion.div>
      )}
    </div>
  );
}

export function MemberAvatar({
  member,
  size = "md",
  className,
  showStatus = true,
}: {
  member: Member;
  size?: "sm" | "md" | "lg";
  className?: string;
  showStatus?: boolean;
}) {
  return (
    <div className={cn("relative inline-flex", className)}>
      <div
        className={cn(
          "rounded-full border-2 border-[var(--background)] overflow-hidden",
          "bg-[var(--muted)]",
          sizeClasses[size],
          "flex items-center justify-center font-medium select-none"
        )}
        style={{
          color: member.color || getColorFromName(member.name),
          backgroundColor: member.avatar ? "transparent" : `${member.color || getColorFromName(member.name)}20`,
        }}
      >
        {member.avatar ? (
          <img
            src={member.avatar}
            alt={member.name}
            className="h-full w-full object-cover"
          />
        ) : (
          member.initials || getInitials(member.name)
        )}
      </div>

      {showStatus && member.status && (
        <span
          className={cn(
            "absolute bottom-0 right-0 rounded-full border-2 border-[var(--background)]",
            statusColors[member.status],
            statusSizeClasses[size]
          )}
        />
      )}
    </div>
  );
}