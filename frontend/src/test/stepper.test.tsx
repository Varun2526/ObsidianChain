import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Stepper, { Step } from "../components/ui/Stepper";

describe("<Stepper />", () => {
  it("renders the initial step content and navigation buttons", () => {
    render(
      <Stepper initialStep={1} backButtonText="Previous" nextButtonText="Continue">
        <Step>
          <div>Step 1 Content</div>
        </Step>
        <Step>
          <div>Step 2 Content</div>
        </Step>
      </Stepper>
    );

    expect(screen.getByText("Step 1 Content")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument();
  });

  it("navigates forward and backward calling onStepChange", () => {
    const onStepChange = vi.fn();
    render(
      <Stepper initialStep={1} onStepChange={onStepChange} backButtonText="Back" nextButtonText="Next">
        <Step>
          <div>Step One</div>
        </Step>
        <Step>
          <div>Step Two</div>
        </Step>
        <Step>
          <div>Step Three</div>
        </Step>
      </Stepper>
    );

    const nextBtn = screen.getByRole("button", { name: "Next" });
    fireEvent.click(nextBtn);

    expect(onStepChange).toHaveBeenCalledWith(2);
    expect(screen.getByText("Step Two")).toBeInTheDocument();

    const backBtn = screen.getByRole("button", { name: "Back" });
    fireEvent.click(backBtn);

    expect(onStepChange).toHaveBeenCalledWith(1);
    expect(screen.getByText("Step One")).toBeInTheDocument();
  });

  it("shows Complete on final step and fires onFinalStepCompleted", () => {
    const onFinalStepCompleted = vi.fn();
    render(
      <Stepper initialStep={2} onFinalStepCompleted={onFinalStepCompleted}>
        <Step>
          <div>Step 1</div>
        </Step>
        <Step>
          <div>Final Step</div>
        </Step>
      </Stepper>
    );

    const completeBtn = screen.getByRole("button", { name: "Complete" });
    expect(completeBtn).toBeInTheDocument();

    fireEvent.click(completeBtn);
    expect(onFinalStepCompleted).toHaveBeenCalledTimes(1);
  });

  it("allows clicking step indicators when not disabled", () => {
    const onStepChange = vi.fn();
    render(
      <Stepper initialStep={1} onStepChange={onStepChange}>
        <Step>
          <div>Step 1</div>
        </Step>
        <Step>
          <div>Step 2</div>
        </Step>
      </Stepper>
    );

    const step2Indicator = screen.getByText("2");
    fireEvent.click(step2Indicator);

    expect(onStepChange).toHaveBeenCalledWith(2);
    expect(screen.getByText("Step 2")).toBeInTheDocument();
  });
});
