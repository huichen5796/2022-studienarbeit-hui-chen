import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SeiteGuideComponent } from './seite-guide.component';

describe('SeiteGuideComponent', () => {
  let component: SeiteGuideComponent;
  let fixture: ComponentFixture<SeiteGuideComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SeiteGuideComponent]
    });
    fixture = TestBed.createComponent(SeiteGuideComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
