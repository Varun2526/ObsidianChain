import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { HookSidebar } from "../components/ui/hook-sidebar";

describe("<HookSidebar />", () => {
  it("renders a list of items with labels and icons", () => {
    const items = [
      { label: "Home", href: "/", icon: <span>⌂</span> },
      { label: "Investigations", href: "/investigations", icon: <span>◫</span> },
      { label: "Settings", href: "/settings", icon: <span>⚙</span> },
    ];

    render(
      <MemoryRouter initialEntries={["/"]}>
        <HookSidebar items={items} label="Main Navigation" />
      </MemoryRouter>
    );

    expect(screen.getByText("Main Navigation")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Investigations")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("marks active item based on current route", () => {
    const items = [
      { label: "Home", href: "/" },
      { label: "Investigations", href: "/investigations" },
    ];

    render(
      <MemoryRouter initialEntries={["/investigations"]}>
        <HookSidebar items={items} />
      </MemoryRouter>
    );

    const activeLink = screen.getByRole("link", { name: "Investigations" });
    expect(activeLink).toHaveAttribute("data-active", "true");
    expect(activeLink).toHaveAttribute("aria-current", "page");

    const homeLink = screen.getByRole("link", { name: "Home" });
    expect(homeLink).toHaveAttribute("data-active", "false");
  });

  it("fires onChange callback on item click", () => {
    const onChange = vi.fn();
    const items = ["Item One", "Item Two", "Item Three"];

    render(
      <MemoryRouter>
        <HookSidebar items={items} onChange={onChange} />
      </MemoryRouter>
    );

    const item2 = screen.getByText("Item Two");
    fireEvent.click(item2);

    expect(onChange).toHaveBeenCalledWith(1);
  });
});
