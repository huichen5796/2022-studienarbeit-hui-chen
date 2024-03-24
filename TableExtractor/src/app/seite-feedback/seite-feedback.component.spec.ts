import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteFeedbackComponent } from './seite-feedback.component';

describe('SeiteFeedbackComponent', () => {
  let component: SeiteFeedbackComponent;
  let fixture: ComponentFixture<SeiteFeedbackComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteFeedbackComponent]
    });
    fixture = TestBed.createComponent(SeiteFeedbackComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
